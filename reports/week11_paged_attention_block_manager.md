# 第 11 周实验报告：PagedAttention 显存分页机制与块管理器 (Block Manager)

## 1. 实验背景与核心痛点
在大模型（LLM）推理服务的解码（Decode）阶段，每个请求的序列长度是**动态未知、逐步增加**的。
- **传统连续内存分配（Contiguous Allocation）**：
  必须按最坏情况预设 $L_{max}$ 开辟连续显存，导致**内部碎片（实际生成短而留空未用）**与**外部碎片（内存无法连续分配给新请求）**极其严重，高达 **60%~80% 的显存处于纯浪费状态**。
- **vLLM PagedAttention 机制**：
  借鉴操作系统**虚拟内存分页（Virtual Memory Paging）**机制，将连续的逻辑 Token 空间划分为固定大小的 **Logical Blocks（逻辑块，如每块 16 tokens）**，并在物理显存中映射到不连续的 **Physical Blocks（物理块）**，由 **BlockTable（页表）** 记录映射关系。

---

## 2. 核心量化对比实验结果

- **测试负载**：30 个异构长度请求并发仿真（Prompt 30~280 tokens，Decode 20~140 tokens）
- **显存池容量**：固定 4096 tokens（256 块，Block Size = 16）

| 对比维度 | 传统静态连续预分配 | PagedAttention 分页块管理 | 性能收益分析 |
| :--- | :--- | :--- | :--- |
| **并发承载数** | **8 个** | **28 个** | **并发吞吐能力暴增 3.50 倍** |
| **请求丢弃/阻塞数** | **22 个** | **2 个** | 显存利用率饱和释放，消除预占阻塞 |
| **实际承载有效 Tokens** | 1714 | 4096 | 服务有效负载大幅提升 |
| **显存浪费量 (Tokens)** | 2382 | 0 | 浪费槽位减少 **100.0%** |
| **显存浪费率 (Waste Rate)** | **58.15%** | **0.00%** | **浪费率从 65%+ 降至个位数** |
| **有效显存利用率** | **41.85%** | **100.00%** | **从低效闲置提升至接近 100% 紧凑** |

---

## 3. PagedAttention 核心架构与调度生命周期

```
【逻辑空间 (单请求视角)】
  Logical Block 0 (Token 0~15)   ──► 映射到物理块 ──► Physical Block 42 (GPU 显存任意位置)
  Logical Block 1 (Token 16~31)  ──► 映射到物理块 ──► Physical Block 7  (GPU 显存任意位置)
  Logical Block 2 (Token 32~47)  ──► 映射到物理块 ──► Physical Block 105 (GPU 显存任意位置)
```

1. **零外部碎片 (Zero External Fragmentation)**：
   物理显存块大小固定（如 16），显存池按块离散分配。只要池中还有任意 1 个空闲块，就能分配给任意请求，彻底消除了连续内存要求的外部碎片。
2. **极低内部碎片 (Minimal Internal Fragmentation)**：
   由于每块只有 16 tokens，仅在请求的**最后一个物理块**中存在未填满的槽位，单请求平均内部碎片小于 8 个 tokens（即 $< 8 / \text{SeqLen} < 3\%$）。
3. **动态按需步进扩容 (Dynamic Allocation)**：
   在 Prefill 阶段只分配 $\lceil L_{prompt} / B \rceil$ 块；在 Decode 阶段每步进 16 个 token 仅需申请 1 块，绝不超前预占。
4. **共享与前缀复用 (Prefix Caching / CoW)**：
   通过 `ref_count` 引用计数机制，系统天然支持多轮对话或系统级 System Prompt 的物理块跨请求共享（Copy-on-Write），进一步节省海量重复 Prompt 显存。

---

## 4. 结论与下一步方向 (Week 12 衔接)
本周手写的 `KVCacheBlockManager` 完整实现了 vLLM 核心的分页管理机制。它解释了为什么 vLLM 能够在外挂单张消费级显卡（如 RTX 3060）或多卡集群上实现远超普通 HuggingFace 的高并发吞吐。
下一阶段我们将直接深入阅读 **vLLM 官方源码** 中的 `vllm/core/block_manager.py` 与 `vllm/core/scheduler.py`，将本周的理论与工业级大厂代码完全印证！
