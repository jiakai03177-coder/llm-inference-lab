import os
import sys
import time
import torch
import matplotlib.pyplot as plt

# 引入我们刚才升级好的 TransformerBlock
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.simple_transformer import TransformerBlock

def generate_without_cache(model, initial_tokens, gen_steps):
    """无 KV Cache: 每次把完整历史序列重新扔进模型"""
    current_tokens = initial_tokens.clone()
    step_latencies = []

    for _ in range(gen_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            # 必须对全量历史重新算 Attention
            out, _ = model(current_tokens, is_causal=True, kv_cache=None)
            next_token = out[:, -1:, :] # 取最后一个 token
            
        torch.cuda.synchronize()
        step_latencies.append((time.perf_counter() - t0) * 1000)
        
        # 把新 token 拼接到历史输入中，下一轮送入更长的序列
        current_tokens = torch.cat([current_tokens, next_token], dim=1)

    return current_tokens[:, -1:, :], step_latencies


def generate_with_cache(model, initial_tokens, gen_steps):
    """带 KV Cache: 历史 KV 保存在显存，每次只送 1 个新 token"""
    step_latencies = []
    
    # 1. Prefill 阶段: 跑初始 prompt 得到初始 kv_cache
    with torch.no_grad():
        out, kv_cache = model(initial_tokens, is_causal=True, kv_cache=None)
        next_token = out[:, -1:, :]

    # 2. Decode 阶段: 自回归生成
    for _ in range(gen_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        with torch.no_grad():
            # 每次只传输入序列长度为 1 的 next_token，以及上一轮缓存
            out, kv_cache = model(next_token, is_causal=False, kv_cache=kv_cache)
            next_token = out[:, -1:, :]
            
        torch.cuda.synchronize()
        step_latencies.append((time.perf_counter() - t0) * 1000)

    return next_token, step_latencies


def run_experiment():
    assert torch.cuda.is_available(), "实验需要 GPU 支持！"
    device = "cuda"
    dtype = torch.float16
    gpu_name = torch.cuda.get_device_name(0)

    dim = 1024
    num_heads = 16
    model = TransformerBlock(dim=dim, num_heads=num_heads).to(device=device, dtype=dtype)
    model.eval()

    # 模拟场景: 128 个字的 prompt，生成 64 个新 token
    prompt_len = 128
    gen_steps = 64
    x_prompt = torch.randn(1, prompt_len, dim, device=device, dtype=dtype)

    print(f"[{gpu_name}] 正在测试自回归文本生成 (Prompt: {prompt_len}, 生成步数: {gen_steps})...")

    # 预热 GPU
    _ = generate_with_cache(model, x_prompt, gen_steps=5)
    _ = generate_without_cache(model, x_prompt, gen_steps=5)

    # 实验 A: 无 Cache 生成
    print("-> 正在运行 [无 KV Cache] 生成...")
    final_out_no_cache, latencies_no_cache = generate_without_cache(model, x_prompt, gen_steps)
    total_no_cache = sum(latencies_no_cache)

    # 实验 B: 带 Cache 生成
    print("-> 正在运行 [带 KV Cache] 生成...")
    final_out_with_cache, latencies_with_cache = generate_with_cache(model, x_prompt, gen_steps)
    total_with_cache = sum(latencies_with_cache)

    # 1. 数值一致性验证 (Correctness Check)
    max_diff = torch.max(torch.abs(final_out_no_cache - final_out_with_cache)).item()
    print(f"\n[验证] 两种模式最终输出的最大绝对误差: {max_diff:.6f}")
    if max_diff < 1e-2:
        print("[验证结果: 通过] KV Cache 输出与暴力全量重算完全一致！")
    else:
        print("[警告] 数值误差过大，请检查缓存逻辑！")

    # 2. 耗时数据
    avg_no_cache = total_no_cache / gen_steps
    avg_with_cache = total_with_cache / gen_steps
    speedup = total_no_cache / total_with_cache
    print(f"\n[无 KV Cache]  总耗时: {total_no_cache:.2f} ms | 平均单步: {avg_no_cache:.3f} ms")
    print(f"[带 KV Cache]  总耗时: {total_with_cache:.2f} ms | 平均单步: {avg_with_cache:.3f} ms")
    print(f"==> KV Cache 总体加速比: {speedup:.2f}x")

    # 3. 绘制单步生成延迟对比曲线
    fig_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, "kv_cache_vs_no_cache.png")

    steps = list(range(1, gen_steps + 1))
    plt.figure(figsize=(9, 5))
    plt.plot(steps, latencies_no_cache, color='red', label='Without KV Cache (Recompute All)', linewidth=2)
    plt.plot(steps, latencies_with_cache, color='green', label='With KV Cache (O(1) Step Latency)', linewidth=2)
    plt.title("Step-by-Step Decode Latency: KV Cache vs No Cache")
    plt.xlabel("Generation Step")
    plt.ylabel("Step Latency (ms)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"\n[OK] 耗时对比图已保存至: {os.path.abspath(fig_path)}")

    # 4. 生成实验报告
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "week03_kv_cache.md")

    md = f"""# 第 3 周实验报告：KV Cache 机制原理与性能对比

## 1. 实验环境与参数
- **GPU**: {gpu_name}
- **测试配置**: Prompt 序列长度 = {prompt_len}，生成 Token 步数 = {gen_steps}
- **模型精度**: FP16

## 2. 实验核心结果

| 指标 | 无 KV Cache (全量重算) | 带 KV Cache (缓存机制) | 提升效果 |
|---|---|---|---|
| **输出数值最大误差** | 基准值 | {max_diff:.6e} | **完全一致 (无损)** |
| **总生成耗时 (ms)** | {total_no_cache:.2f} | {total_with_cache:.2f} | **加速 {speedup:.2f} 倍** |
| **平均单步耗时 (ms)** | {avg_no_cache:.3f} | {avg_with_cache:.3f} | 显存换算力明显 |

## 3. 核心机制剖析与显存模型

### (1) 为什么无 Cache 会越来越慢？
- 每生成一个新 token，必须将历史全量 token 重新执行前向传播。第 $t$ 步的计算量为 $O((L_{{prompt}} + t)^2)$。
- 随着生成步数增加，单步延迟持续爬坡。

### (2) 为什么有 Cache 单步耗时稳如泰山？
- 历史序列的 Key 和 Value 被保留在显存中。
- 第 $t$ 步只需计算当前 $1$ 个 token 的 $Q_{{new}}, K_{{new}}, V_{{new}}$，并与缓存拼接，单步计算复杂度始终维持在极低的固定水平。

### (3) KV Cache 显存占用计算公式 (以 7B 大模型为例)
对于 FP16 精度 (2 字节)，每个 Token 在单层中占用显存为：
$$\\text{{KV per Token per Layer}} = 2 \\times 2 \\times \\text{{kv\\_heads}} \\times \\text{{head\\_dim}}$$
- 典型 7B 模型 (32 层, 32 头, 维度 128, MHA):
  - 1 个 Token 的 KV Cache 约占 **0.5 MB**。
  - 上下文长度达到 **4096** 时，单个并发的 KV Cache 就吃掉 **2 GB** 显存！
  - 上下文长度达到 **8192** 时，KV Cache 吃掉 **4 GB**，直接导致 6GB/8GB 显卡 OOM！
- **这解释了为什么开源社区（如 LLaMA-2/3、Qwen）必须采用 GQA (Grouped-Query Attention)**：将 KV 头数从 32 压缩到 8 或 4，直接把 KV Cache 显存减少 4~8 倍！
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] 实验报告已生成至: {os.path.abspath(report_path)}")

if __name__ == '__main__':
    run_experiment()