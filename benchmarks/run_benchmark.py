import os
import sys
import time
import argparse
import torch

# 引入我们的模型
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.simple_transformer import TransformerBlock

def parse_args():
    parser = argparse.ArgumentParser(description="LLM 推理性能 Benchmark 工具")
    parser.add_argument("--batch-size", type=int, default=1, help="并发批处理大小 (Batch Size)")
    parser.add_argument("--input-len", type=int, default=256, help="输入 Prompt 长度")
    parser.add_argument("--output-len", type=int, default=64, help="生成 Token 长度")
    parser.add_argument("--dim", type=int, default=1024, help="隐藏层维度")
    parser.add_argument("--heads", type=int, default=16, help="注意力头数")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"], help="数据精度")
    return parser.parse_args()

def run_benchmark():
    args = parse_args()
    assert torch.cuda.is_available(), "本测试必须在 GPU 环境下运行！"
    device = "cuda"
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 60)
    print(f"🚀 启动 LLM 推理性能 Benchmark")
    print(f"设备: {gpu_name}")
    print(f"配置: Batch Size={args.batch_size}, Input Len={args.input_len}, Output Len={args.output_len}, Dtype={args.dtype}")
    print("=" * 60)

    # 1. 实例化模型
    model = TransformerBlock(dim=args.dim, num_heads=args.heads).to(device=device, dtype=dtype)
    model.eval()

    # 2. 准备虚拟输入
    x_input = torch.randn(args.batch_size, args.input_len, args.dim, device=device, dtype=dtype)

    # 3. 预热 (Warmup)
    print("正在预热 GPU...")
    with torch.no_grad():
        out, cache = model(x_input, is_causal=True)
        for _ in range(3):
            out, cache = model(out[:, -1:, :], is_causal=False, kv_cache=cache)
    torch.cuda.synchronize()

    # 4. 重置显存峰值统计
    torch.cuda.reset_peak_memory_stats()

    # 5. 测试阶段 1: Prefill (首字延迟 TTFT)
    print("正在测试 Prefill 阶段...")
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    
    with torch.no_grad():
        out, kv_cache = model(x_input, is_causal=True, kv_cache=None)
        next_token = out[:, -1:, :]
        
    torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t_start) * 1000

    # 6. 测试阶段 2: Decode (后续生成过程)
    print(f"正在测试 Decode 阶段 (自回归生成 {args.output_len - 1} 步)...")
    decode_latencies = []
    
    for _ in range(args.output_len - 1):
        torch.cuda.synchronize()
        t_step = time.perf_counter()
        
        with torch.no_grad():
            out, kv_cache = model(next_token, is_causal=False, kv_cache=kv_cache)
            next_token = out[:, -1:, :]
            
        torch.cuda.synchronize()
        decode_latencies.append((time.perf_counter() - t_step) * 1000)

    # 7. 计算各项黄金指标
    t_total_ms = ttft_ms + sum(decode_latencies)
    tpot_ms = sum(decode_latencies) / len(decode_latencies) if decode_latencies else 0.0
    
    # 系统总吞吐量: 这一批次生成的全部 Token 数 / 总生成耗时(秒)
    total_tokens_generated = args.batch_size * args.output_len
    throughput_tokens_per_s = total_tokens_generated / (t_total_ms / 1000)
    
    # 显存峰值 (MB)
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # 8. 格式化打印控制台
    print("\n" + "-" * 40)
    print(" 📊 Benchmark 评测结果报告")
    print("-" * 40)
    print(f"首字延迟 (TTFT)        : {ttft_ms:.2f} ms")
    print(f"单字解码延迟 (TPOT)    : {tpot_ms:.2f} ms")
    print(f"端到端总延迟 (E2E)      : {t_total_ms:.2f} ms")
    print(f"系统总吞吐 (Throughput): {throughput_tokens_per_s:.2f} tokens/s")
    print(f"显存占用峰值 (Peak Mem) : {peak_mem_mb:.2f} MB")
    print("-" * 40)

    # 9. 追加保存到报告 week04_benchmark.md
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "week04_benchmark.md")
    
    # 如果文件不存在，先写表头
    need_header = not os.path.exists(report_path)
    with open(report_path, "a", encoding="utf-8") as f:
        if need_header:
            f.write(f"# 第 4 周实验报告：推理 Benchmark 与性能指标体系\n\n")
            f.write(f"- **硬件设备**: {gpu_name}\n\n")
            f.write(f"| Batch Size | Input Len | Output Len | TTFT (ms) | TPOT (ms) | E2E (ms) | 吞吐量 (tokens/s) | 显存峰值 (MB) |\n")
            f.write(f"|---|---|---|---|---|---|---|---|\n")
        f.write(f"| {args.batch_size} | {args.input_len} | {args.output_len} | {ttft_ms:.2f} | {tpot_ms:.2f} | {t_total_ms:.2f} | {throughput_tokens_per_s:.2f} | {peak_mem_mb:.2f} |\n")

    print(f"[OK] 结果已成功追加至: {os.path.abspath(report_path)}")

if __name__ == '__main__':
    run_benchmark()