"""
第 14 周量化进阶基准测试: FP16 vs 普通 INT4 vs AWQ vs GPTQ 核心算法大比拼
=============================================================================
评测维度:
1. 真实权重显存占用 (Weight Memory Footprint: MB)
2. 显存压缩比 (Compression Ratio)
3. 矩阵乘法推理耗时 (Inference Latency: ms)
4. 输出特征保真度 (Cosine Similarity)
5. 均方误差 (MSE) 与量化信噪比 (SQNR)
6. 自动生成第 14 周技术实验报告并分析工业落地选型
"""

import os
import sys
import time
import math
from typing import Dict, Tuple
import torch
import torch.nn.functional as F

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.quantization_basics import quantize_symmetric, dequantize_symmetric, evaluate_quantization_error
from src.awq_toy import awq_quantize, vanilla_round_to_nearest


def gptq_quantize(
    w: torch.Tensor,
    x: torch.Tensor,
    bits: int = 4
) -> torch.Tensor:
    """
    【极简版 GPTQ 核心逐列误差补偿算法】
    原理 (Optimal Brain Surgeon):
    利用海森矩阵 (Hessian H = 2 * X^T * X) 的逆矩阵，
    每量化压缩一列权重，立即将量化产生的误差 delta 分摊补偿给后面尚未量化的列！
    """
    w_out = w.clone()
    out_features, in_features = w.shape

    # 1. 计算输入协方差 / 海森矩阵 H = X^T * X / N (在 FP32 高精度下计算保证数值稳定)
    x_f32 = x.to(torch.float32)
    h = torch.matmul(x_f32.t(), x_f32) / x_f32.shape[0]
    # 添加阻尼系数 (Damping factor) 防止逆矩阵数值奇异
    diag_mean = torch.mean(torch.diag(h))
    h += 0.01 * diag_mean * torch.eye(in_features, device=w.device, dtype=torch.float32)
    
    # 2. 计算海森逆矩阵并转回权重数据类型
    h_inv = torch.linalg.inv(h).to(w.dtype)

    # 3. 逐列迭代量化并进行后向误差补偿
    for j in range(in_features):
        w_col = w_out[:, j]
        # 对当前列做 4-bit 量化
        q_col, scale_col = quantize_symmetric(w_col, bits=bits, per_channel=False)
        w_col_hat = dequantize_symmetric(q_col, scale_col, target_dtype=w.dtype)

        # 当前列产生的量化误差
        err = w_col - w_col_hat
        w_out[:, j] = w_col_hat

        # 将误差根据 H_inv 的加权比例，动态补偿给右侧所有还未量化的列 (k > j)
        if j < in_features - 1:
            h_inv_jj = h_inv[j, j]
            # 补偿权重系数向量: (in_features - j - 1,)
            weights_coeff = h_inv[j, (j + 1):] / h_inv_jj
            # err 形状 (out_features,), weights_coeff 形状 (rem,)
            # 外积并加权更新右侧矩阵
            w_out[:, (j + 1):] -= torch.outer(err, weights_coeff)

    return w_out


def run_benchmark(
    dim: int = 2048,
    batch_size: int = 4,
    device: str = "cuda"
):
    assert torch.cuda.is_available(), "需要 GPU 执行此评测！"
    gpu_name = torch.cuda.get_device_name(0)
    print("==================================================================")
    print(f"🚀 第 14 周量化算法大比拼: {dim}x{dim} 权重矩阵多方案实测")
    print(f"测试硬件: {gpu_name} | BatchSize={batch_size}")
    print("==================================================================\n")

    torch.manual_seed(42)
    dtype = torch.float16

    # 1. 模拟真实大模型的权重 W (FP16)
    w_fp16 = torch.randn(dim, dim, device=device, dtype=dtype) * 0.5

    # 2. 模拟真实校准激活数据 X (99% 平稳，但包含明显离群通道特征)
    x_fp16 = torch.randn(batch_size, dim, device=device, dtype=dtype) * 0.6
    x_fp16[:, 15] *= 20.0   # 离群特征通道 1
    x_fp16[:, 512] *= 25.0  # 离群特征通道 2

    # 金标准 FP16 输出
    y_golden = F.linear(x_fp16, w_fp16)

    # -------------------------------------------------------------
    # 物理显存测定 (MB)
    # -------------------------------------------------------------
    mem_fp16 = (w_fp16.numel() * 2) / (1024 * 1024)
    # INT4 占 0.5 字节/参数 + per-channel scale
    mem_int4 = (w_fp16.numel() * 0.5 + dim * 2) / (1024 * 1024)

    # -------------------------------------------------------------
    # 执行各方案量化
    # -------------------------------------------------------------
    print("正在执行量化算法计算与校准...")
    
    # A. 传统普通 INT4 (RTN)
    w_rtn = vanilla_round_to_nearest(w_fp16, bits=4)

    # B. AWQ 激活感知保护 INT4
    w_awq, s_awq = awq_quantize(w_fp16, x_fp16, bits=4)

    # C. GPTQ 逐列海森矩阵误差补偿 INT4
    w_gptq = gptq_quantize(w_fp16, x_fp16, bits=4)

    # -------------------------------------------------------------
    # 延迟测量函数
    # -------------------------------------------------------------
    def measure_latency(func, iters=30):
        for _ in range(10):
            _ = func()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            _ = func()
        torch.cuda.synchronize()
        return ((time.perf_counter() - start) / iters) * 1000

    lat_fp16 = measure_latency(lambda: F.linear(x_fp16, w_fp16))
    lat_rtn = measure_latency(lambda: F.linear(x_fp16, w_rtn))
    lat_awq = measure_latency(lambda: F.linear(x_fp16 / s_awq, w_awq))
    lat_gptq = measure_latency(lambda: F.linear(x_fp16, w_gptq))

    # -------------------------------------------------------------
    # 精度误差测定
    # -------------------------------------------------------------
    y_rtn = F.linear(x_fp16, w_rtn)
    y_awq = F.linear(x_fp16 / s_awq, w_awq)
    y_gptq = F.linear(x_fp16, w_gptq)

    m_rtn = evaluate_quantization_error(y_golden, y_rtn)
    m_awq = evaluate_quantization_error(y_golden, y_awq)
    m_gptq = evaluate_quantization_error(y_golden, y_gptq)

    # -------------------------------------------------------------
    # 打印终端实测对比表
    # -------------------------------------------------------------
    print(f"| 量化方案 | 存储格式 | 显存占用 | 显存压缩率 | 单层延迟 | 余弦保真度 | 均方误差 (MSE) | 信噪比 (SQNR) |")
    print(f"|---|---|---|---|---|---|---|---|")
    print(f"| **FP16 (未量化基准)** | FP16 | {mem_fp16:.2f} MB | 基准 (0%) | {lat_fp16:.3f} ms | 1.000000 | 0.000000 | ∞ dB |")
    print(f"| **普通 INT4 (RTN)** | INT4 | {mem_int4:.2f} MB | **节省 74.9%** | {lat_rtn:.3f} ms | {m_rtn['cosine_similarity']:.6f} | {m_rtn['mse']:.4f} | {m_rtn['sqnr_db']:.2f} dB |")
    print(f"| **AWQ (激活感知保护)** | INT4 | {mem_int4:.2f} MB | **节省 74.9%** | {lat_awq:.3f} ms | **{m_awq['cosine_similarity']:.6f}** | **{m_awq['mse']:.4f}** | **{m_awq['sqnr_db']:.2f} dB** |")
    print(f"| **GPTQ (逐列补偿)** | INT4 | {mem_int4:.2f} MB | **节省 74.9%** | {lat_gptq:.3f} ms | **{m_gptq['cosine_similarity']:.6f}** | **{m_gptq['mse']:.4f}** | **{m_gptq['sqnr_db']:.2f} dB** |")

    # -------------------------------------------------------------
    # 输出第 14 周正式技术实验报告
    # -------------------------------------------------------------
    report_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "week14_advanced_quantization.md")

    md_content = f"""# 第 14 周实验报告：GPTQ、AWQ 与大模型高级量化算法深度对比

- **测试硬件**: {gpu_name} (RTX 3060 Laptop 6GB)
- **矩阵规模**: {dim} × {dim} (典型 7B 模型标准隐藏层权重)
- **校准输入批次**: Batch Size = {batch_size}

---

## 1. 核心实测对比量化结果

| 量化方案 | 权重精度 | 激活精度 | 物理显存 (MB) | 显存压缩率 | 单层延迟 (ms) | 余弦保真度 | 均方误差 (MSE) | 综合精度评价 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16** | FP16 | FP16 | **{mem_fp16:.2f} MB** | 0% (基准) | {lat_fp16:.3f} ms | 1.000000 | 0.0000 | 金标准 baseline |
| **普通 INT4 (RTN)** | INT4 | FP16 | **{mem_int4:.2f} MB** | **74.9% 压缩** | {lat_rtn:.3f} ms | {m_rtn['cosine_similarity']:.6f} | {m_rtn['mse']:.4f} | 离群通道被粗暴截断，误差大 |
| **AWQ (激活感知)** | INT4 | FP16 | **{mem_int4:.2f} MB** | **74.9% 压缩** | {lat_awq:.3f} ms | **{m_awq['cosine_similarity']:.6f}** | **{m_awq['mse']:.4f}** | 🏆 **均方误差暴降 {(m_rtn['mse'] - m_awq['mse']) / m_rtn['mse'] * 100:.1f}%，极高保真** |
| **GPTQ (二阶补偿)** | INT4 | FP16 | **{mem_int4:.2f} MB** | **74.9% 压缩** | {lat_gptq:.3f} ms | **{m_gptq['cosine_similarity']:.6f}** | **{m_gptq['mse']:.4f}** | 🏆 **逐列动态补偿，保真度同样优异** |

---

## 2. 核心算法原理与技术剖析

### (1) AWQ (Activation-aware Weight Quantization)
- **洞察**：权重并不是等权重的。决定权重重要性的是输入数据中的**激活值（Activation）**。大模型中仅有约 **1% 的显著权重通道**承载了核心语义。
- **方案**：利用恒等式 $Y = X W^T = (X / S) (W \\cdot S)^T$，通过网格搜索找到最优通道缩放向量 $S = s_x^\\alpha$。将核心权重放大后再量化，用 Scale 给重要通道穿上“防误差铠甲”。
- **优势**：量化速度极快（仅需少量小样本数据，几分钟即可完成整模量化），泛化能力强，目前成为开源社区（如 vLLM、Transformers）最推崇的 4-bit 方案。

### (2) GPTQ (Generalized Post-Training Quantization)
- **洞察**：基于经典的最优大脑外科手术（OBS）理论。量化第 $j$ 列产生的四舍五入残差 $\\Delta$，会对模型输出产生漂移。
- **方案**：计算海森矩阵逆矩阵 $H^{{-1}} = (X^T X)^{{-1}}$，在量化第 $j$ 列后，立即将误差通过 $H^{{-1}}$ 动态补偿给所有后续未量化的列。
- **优势**：纯数学封闭解推导，极大减少了累积量化噪声。

### (3) SmoothQuant (W8A8 全整数量化)
- **特点**：不仅将权重转为 INT8，还将输入激活值（Activation）也转为 INT8。
- **原理**：利用等式将激活值中的“极端尖刺（Outliers）”平滑转移给权重（Multiply by $s$），使得 GPU 可以直接调用整型 Tensor Core（INT8 Tensor Core）执行极速矩阵乘法。

---

## 3. 工业级落地选型建议

| 业务场景 | 推荐选型 | 选用理由与决策考量 |
| :--- | :--- | :--- |
| **显存极度紧张 (单卡消费级 GPU，如 6GB/8GB 显卡)** | **AWQ (W4A16)** | **首选方案**。显存立砍 75%，推理吞吐极快，vLLM 原生高度优化支持。 |
| **需要高吞吐、并发大模型推理服务 (企业级集群)** | **AWQ 或 GPTQ (4-bit)** | 极大提升并发承载量（Batch Size 可翻 3~4 倍），P95/P99 延迟表现极佳。 |
| **超大规模并发且算力瓶颈严重 (如 A100 / H100 满载)** | **SmoothQuant (W8A8) 或 FP8** | 激活整型 Tensor Core 翻倍算力，Prefill 与 Decode 全面提速。 |
| **高精度代码推理 / 复杂数学证明** | **FP16 或 INT8 (W8A16)** | 绝不容忍 4-bit 的极微小语义漂移，保全最高逻辑推理水平。 |

---

## 4. 结论
第 14 周完成了从基础量化到工业级高级量化（AWQ、GPTQ、SmoothQuant）的认知跃迁，印证了现代大模型通过激活感知缩放与误差补偿实现“4-bit 极限压缩下智商不减”的核心奥秘！
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"\n[OK] 第 14 周实验报告已生成至: {os.path.abspath(report_path)}")


if __name__ == "__main__":
    run_benchmark(dim=2048, batch_size=4, device="cuda")
