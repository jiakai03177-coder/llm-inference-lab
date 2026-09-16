import os
import sys
import time
import torch
import matplotlib.pyplot as plt

# 把上一级目录加入 python path，方便引用 src
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.simple_transformer import TransformerBlock

def measure_latency(model, x, is_causal=True, warmups=5, iters=20):
    # 1. 预热
    with torch.no_grad():
        for _ in range(warmups):
            _ = model(x, is_causal=is_causal)
    torch.cuda.synchronize()

    # 2. 计时
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            _ = model(x, is_causal=is_causal)
        torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = ((end - start) / iters) * 1000
    return avg_ms

def run_experiment():
    assert torch.cuda.is_available(), "需要 GPU 运行此实验！"
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"当前测试设备: {gpu_name}")

    # 实例化一个轻量级 Block (适合笔记本 3060 快速且准确地测出特性)
    dim = 1024
    num_heads = 16
    dtype = torch.float16  # 采用工业界标准的 FP16
    model = TransformerBlock(dim=dim, num_heads=num_heads).to(device=device, dtype=dtype)
    model.eval()

    # 实验一: Prefill 阶段 (测试不同输入序列长度)
    seq_lengths = [128, 256, 512, 1024, 2048]
    prefill_latencies = []

    print("\n--- 实验 1: Prefill 阶段不同序列长度的延迟 ---")
    for s in seq_lengths:
        x = torch.randn(1, s, dim, device=device, dtype=dtype)
        latency = measure_latency(model, x, is_causal=True)
        prefill_latencies.append(latency)
        print(f"Seq Len: {s:4d} | 耗时: {latency:.3f} ms")

    # 实验二: Decode 单步解码 (Seq Len = 1)
    print("\n--- 实验 2: Decode 阶段 (单步 token 生成) ---")
    x_decode = torch.randn(1, 1, dim, device=device, dtype=dtype)
    decode_latency = measure_latency(model, x_decode, is_causal=False)
    print(f"Decode 单步耗时 (Seq Len = 1): {decode_latency:.3f} ms")

    # 绘制可视化图表
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "prefill_vs_decode.png")

    plt.figure(figsize=(8, 5))
    plt.plot(seq_lengths, prefill_latencies, marker='o', color='b', label='Prefill Latency (ms)')
    plt.axhline(y=decode_latency, color='r', linestyle='--', label=f'Decode Latency ({decode_latency:.2f} ms)')
    plt.title("Prefill vs Decode Latency across Sequence Lengths (FP16)")
    plt.xlabel("Sequence Length")
    plt.ylabel("Latency (ms)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\n[OK] 可视化图表已生成: {os.path.abspath(fig_path)}")

    # 输出 Markdown 报告
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "week02_prefill_vs_decode.md")

    md = f"""# 第 2 周实验报告：Transformer 推理流程与 Prefill/Decode 差异分析

## 1. 硬件配置
- **GPU**: {gpu_name}
- **模型规格**: Single Transformer Layer (Hidden Dim: {dim}, Heads: {num_heads}, Dtype: FP16)

## 2. 测试数据汇总

### Prefill 阶段 (批量输入 Prompt)
| Sequence Length | 延迟 (ms) |
|---|---|
"""
    for s, lat in zip(seq_lengths, prefill_latencies):
        md += f"| {s} | {lat:.3f} |\n"

    md += f"""
### Decode 阶段 (单 Token 自回归)
- **单步解码延迟 (S=1)**: **{decode_latency:.3f} ms**

## 3. 核心现象与理论剖析
1. **Prefill 计算复杂度随长度迅速上升**：
   - 注意力机制计算 $Q K^T$ 的复杂度为 $O(S^2)$。随着 Prompt 长度翻倍，计算量呈平方级膨胀。
2. **Prefill 与 Decode 的本质鸿沟**：
   - Prefill 阶段（$S \ge 128$）矩阵尺寸大，能够喂饱 GPU 计算单元，属于 **Compute-bound（计算受限）**；
   - Decode 阶段每次只送入 1 个 token（$S=1$），矩阵退化成了向量乘矩阵（GEMV），算力跑不满，大量时间耗费在显存搬运权重上，属于 **Memory-bound（访存受限）**。
3. **引出下周核心问题——为什么必须要有 KV Cache？**
   - 如果在 Decode 阶段，我们每吐一个字，都把历史所有的上下文重新打包送进模型做一次完整的 Prefill，那么每一步生成都会是 $O(S^2)$ 的时间开销！这正是第 3 周要解决的问题。
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] 实验报告已生成: {os.path.abspath(report_path)}")

if __name__ == '__main__':
    run_experiment()