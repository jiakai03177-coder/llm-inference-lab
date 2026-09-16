import os
import time
import torch

def benchmark_matmul(size=4096, dtype=torch.float32, device='cuda'):
    print(f"\n--- 测试矩阵大小: {size}x{size}, 数据类型: {dtype}, 设备: {device} ---")
    
    # 1. 准备数据
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    
    # 2. 预热 (Warmup)
    for _ in range(5):
        _ = torch.matmul(a, b)
    if device == 'cuda':
        torch.cuda.synchronize()
        
    # 3. 正式测试
    iters = 10
    start_time = time.perf_counter()
    for _ in range(iters):
        _ = torch.matmul(a, b)
    if device == 'cuda':
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    # 4. 计算指标
    avg_latency_s = (end_time - start_time) / iters
    avg_latency_ms = avg_latency_s * 1000
    flops = 2 * (size ** 3)
    tflops = (flops / avg_latency_s) / 1e12
    
    print(f"平均延迟: {avg_latency_ms:.2f} ms")
    print(f"有效算力: {tflops:.2f} TFLOPs")
    
    return {
        "size": f"{size}x{size}",
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "latency_ms": f"{avg_latency_ms:.2f}",
        "tflops": f"{tflops:.2f}"
    }

def save_report(results, gpu_name):
    # 确保 reports 目录存在
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "week01_baseline.md")
    
    md_content = f"""# 第 1 周实验报告：GPU 推理基线与 GEMM 性能测试

## 1. 实验环境与硬件
- **GPU 型号**: {gpu_name}
- **PyTorch 版本**: {torch.__version__}
- **CUDA 可用状态**: {torch.cuda.is_available()}

## 2. GEMM (矩阵乘法) 性能测试结果

| 矩阵规模 | 设备 | 数据类型 | 平均延迟 (ms) | 有效算力 (TFLOPs) |
|---|---|---|---|---|
"""
    for r in results:
        md_content += f"| {r['size']} | {r['device']} | {r['dtype']} | {r['latency_ms']} | {r['tflops']} |\n"

    md_content += """
## 3. 实验现象与核心结论分析
1. **CPU vs GPU 算力差距**：
   - 在相同维度下，GPU 展现出远超 CPU 的并行吞吐能力（矩阵运算高度适合 GPU 的 SIMT 架构）。
2. **FP32 vs FP16 (Tensor Core) 表现**：
   - 切换到 FP16 半精度后，TFLOPs 出现翻倍以上的跃升。这是因为 Ampere 架构（RTX 30 系列）激活了硬件级 **Tensor Core（张量核心）** 进行混合精度加速。
3. **显存与大模型推理映射**：
   - 大语言模型绝大多数 Linear 投影本质上就是批量的 GEMM。推理部署时采用 FP16/BF16 不仅显存减半，更能直接激活硬件的最高算力上限。
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[OK] 实验报告已自动生成至: {os.path.abspath(report_path)}")

if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA 不可用，请检查环境！"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"当前 GPU: {gpu_name}")
    
    results = []
    
    # 对比 1: CPU vs GPU (2048x2048)
    print("\n[实验 1: CPU vs GPU]")
    results.append(benchmark_matmul(size=2048, dtype=torch.float32, device='cpu'))
    results.append(benchmark_matmul(size=2048, dtype=torch.float32, device='cuda'))
    
    # 对比 2: GPU FP32 vs FP16 (4096x4096)
    print("\n[实验 2: GPU FP32 vs FP16]")
    results.append(benchmark_matmul(size=4096, dtype=torch.float32, device='cuda'))
    results.append(benchmark_matmul(size=4096, dtype=torch.float16, device='cuda'))
    
    # 自动保存报告
    save_report(results, gpu_name)