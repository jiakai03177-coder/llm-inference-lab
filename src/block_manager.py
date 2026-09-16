"""
PagedAttention 显存块管理器 (Block Manager) 模块
===================================================
实现基于操作系统虚拟内存分页机制的 KV Cache 管理器。
核心解决传统大模型推理中连续显存预分配导致的巨额内部/外部碎片与显存浪费。

对应 vLLM 核心设计:
- PhysicalTokenBlock: 物理显存块 (固定容量，通常为 16 或 32 个 tokens)
- BlockTable: 逻辑块 (Logical Block) 到物理块 (Physical Block) 的页表映射
- KVCacheBlockManager: 显存池生命周期调度器 (分配、按需追加、释放、复用)
"""

import math
from typing import List, Dict, Optional, Tuple


class PhysicalTokenBlock:
    """
    物理显存块 (Physical Token Block)
    表示在 GPU 显存池中已开辟的固定大小的物理存储单元。
    """
    def __init__(self, block_id: int, block_size: int = 16):
        self.block_id = block_id          # 物理块全局唯一编号 (0, 1, 2, ...)
        self.block_size = block_size      # 该物理块最大能容纳的 token 数量 (默认 16)
        self.num_tokens = 0               # 当前块内已使用的 token 插槽数量
        self.ref_count = 0                # 引用计数 (用于多轮对话、Prefix Caching 或束搜索 Fork)

    @property
    def is_full(self) -> bool:
        """物理块是否已被填满"""
        return self.num_tokens >= self.block_size

    @property
    def has_space(self) -> bool:
        """物理块是否还有空余插槽"""
        return self.num_tokens < self.block_size

    @property
    def free_slots(self) -> int:
        """块内剩余未使用的插槽数 (内部碎片来源)"""
        return self.block_size - self.num_tokens

    def append_token(self) -> bool:
        """向当前物理块追加写入 1 个 token 插槽"""
        if self.is_full:
            return False
        self.num_tokens += 1
        return True

    def reset(self):
        """重置物理块状态并归还显存池"""
        self.num_tokens = 0
        self.ref_count = 0

    def __repr__(self):
        return f"Block(ID={self.block_id}, used={self.num_tokens}/{self.block_size}, refs={self.ref_count})"


class KVCacheBlockManager:
    """
    KV Cache 分页块管理器 (Block Manager)
    负责统一调度全系统的物理显存块，维护各活跃请求的 BlockTable (页表)。
    """
    def __init__(self, num_total_blocks: int, block_size: int = 16):
        """
        :param num_total_blocks: 系统预先分配的总物理块数 (由可用 GPU 显存大小决定)
        :param block_size: 每个物理块的 token 槽位数
        """
        self.num_total_blocks = num_total_blocks
        self.block_size = block_size

        # 初始化物理块对象池
        self.all_blocks: List[PhysicalTokenBlock] = [
            PhysicalTokenBlock(block_id=i, block_size=block_size)
            for i in range(num_total_blocks)
        ]

        # 空闲物理块池 (Free List / Stack)，初始时所有物理块皆空闲
        self.free_blocks: List[PhysicalTokenBlock] = list(reversed(self.all_blocks))

        # 页表映射: request_id -> List[PhysicalTokenBlock] (按逻辑顺序排列)
        self.block_tables: Dict[str, List[PhysicalTokenBlock]] = {}

        # 跟踪每个请求的累计 token 长度 (Prompt + 已生成的 Decode tokens)
        self.request_token_counts: Dict[str, int] = {}

    def get_num_free_blocks(self) -> int:
        """当前可用空闲物理块数"""
        return len(self.free_blocks)

    def can_allocate(self, num_tokens: int) -> bool:
        """检查显存池是否有足够的空闲块来容纳指定数量的 tokens"""
        needed_blocks = math.ceil(num_tokens / self.block_size) if num_tokens > 0 else 1
        return self.get_num_free_blocks() >= needed_blocks

    def allocate(self, request_id: str, num_prompt_tokens: int) -> List[int]:
        """
        【Prefill 阶段】：为一个新请求分配初始物理显存块。
        按需计算最小必要块数，绝不超额预占，实现零外部碎片。
        
        :return: 分配给该请求的物理块 ID 列表 (即初始 BlockTable)
        """
        if request_id in self.block_tables:
            raise ValueError(f"请求 {request_id} 已存在，不能重复分配！")

        needed_blocks = math.ceil(num_prompt_tokens / self.block_size) if num_prompt_tokens > 0 else 1
        if self.get_num_free_blocks() < needed_blocks:
            raise MemoryError(
                f"显存不足！请求需 {needed_blocks} 块，当前仅剩 {self.get_num_free_blocks()} 空闲块。"
            )

        assigned_blocks: List[PhysicalTokenBlock] = []
        remaining_tokens = num_prompt_tokens

        for _ in range(needed_blocks):
            block = self.free_blocks.pop()
            block.ref_count = 1
            # 填入 token: 前面的块全满，最后一块填入剩余 token
            slots_to_fill = min(remaining_tokens, self.block_size)
            block.num_tokens = slots_to_fill
            remaining_tokens -= slots_to_fill

            assigned_blocks.append(block)

        self.block_tables[request_id] = assigned_blocks
        self.request_token_counts[request_id] = num_prompt_tokens

        return [b.block_id for b in assigned_blocks]

    def can_append_slot(self, request_id: str) -> bool:
        """检查当前请求在下一步 decode 时是否能容纳新的 1 个 token"""
        if request_id not in self.block_tables:
            return False

        last_block = self.block_tables[request_id][-1]
        # 如果最后一个物理块还有空槽位，不需要申请新块，直接可用
        if last_block.has_space:
            return True
        # 如果最后一个物理块已满，必须有至少 1 个空闲块可用
        return self.get_num_free_blocks() >= 1

    def append_slot(self, request_id: str) -> Tuple[bool, int]:
        """
        【Decode 阶段】：当前请求每生成 1 个新 token 时调用。
        如果当前块未满，直接在块内追加；如果满，按需从池中新开辟 1 个物理块挂入页表。
        
        :return: (成功标记, 当前写入的目标物理块 ID)
        """
        if request_id not in self.block_tables:
            raise KeyError(f"未找到请求 ID: {request_id}")

        last_block = self.block_tables[request_id][-1]

        if last_block.has_space:
            last_block.append_token()
            self.request_token_counts[request_id] += 1
            return True, last_block.block_id
        else:
            # 最后一个物理块已满，必须新开辟一个物理块
            if self.get_num_free_blocks() == 0:
                # 显存耗尽 (OOM)，需要触发抢占 (Preemption) 或等待
                return False, -1

            new_block = self.free_blocks.pop()
            new_block.ref_count = 1
            new_block.append_token()  # 放入第 1 个 token

            self.block_tables[request_id].append(new_block)
            self.request_token_counts[request_id] += 1
            return True, new_block.block_id

    def free(self, request_id: str):
        """
        【请求结束阶段】：请求生成完成或被中断时，立即归还其所有物理块。
        支持引用计数：如果某物理块被共享（如 Prefix Cache），仅减引用计数；当引用为 0 时归还空闲池。
        """
        if request_id not in self.block_tables:
            return

        blocks = self.block_tables.pop(request_id)
        del self.request_token_counts[request_id]

        for block in blocks:
            block.ref_count -= 1
            if block.ref_count <= 0:
                block.reset()
                self.free_blocks.append(block)

    def get_block_table(self, request_id: str) -> List[int]:
        """获取指定请求的物理块号映射列表，供底层 PagedAttention Kernel 寻址"""
        if request_id not in self.block_tables:
            return []
        return [b.block_id for b in self.block_tables[request_id]]

    def get_memory_stats(self) -> dict:
        """
        统计当前显存系统的实时运行指标与碎片率
        """
        used_blocks = self.num_total_blocks - len(self.free_blocks)
        total_tokens_stored = sum(self.request_token_counts.values())
        allocated_capacity = used_blocks * self.block_size

        # 内部碎片 = 已分配的物理块总插槽数 - 实际存储的 tokens 数量
        internal_frag_tokens = allocated_capacity - total_tokens_stored
        internal_frag_rate = (internal_frag_tokens / allocated_capacity) if allocated_capacity > 0 else 0.0

        return {
            "total_blocks": self.num_total_blocks,
            "free_blocks": len(self.free_blocks),
            "used_blocks": used_blocks,
            "utilization_rate": (used_blocks / self.num_total_blocks) if self.num_total_blocks > 0 else 0.0,
            "active_requests": len(self.block_tables),
            "total_tokens_stored": total_tokens_stored,
            "allocated_capacity_tokens": allocated_capacity,
            "internal_frag_tokens": internal_frag_tokens,
            "internal_frag_rate": internal_frag_rate,
            "external_frag_rate": 0.0  # Paged 机制的外部碎片在数学上为 0
        }

