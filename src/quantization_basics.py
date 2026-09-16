"""
低精度浮点与模型量化基础 (Quantization Basics) 模块
===================================================
实现大模型量化中最核心的数学变换:
1. 对称量化 (Symmetric AbsMax Quantization)
2. 非对称量化 (Asymmetric MinMax Quantization)
3. 逐张量 (Per-Tensor) 与 逐通道 (Per-Channel) 粒度控制
4. INT8 (8-bit) 与 INT4 (4-bit) 压缩与反量化重构
5. 量化误差量化分析指标 (Cosine Similarity, MSE, Max Error, SQNR)
"""

import math
from typing import Tuple, Dict
import torch
import torch.nn.functional as F


def quantize_symmetric(
    x: torch.Tensor,
    bits: int = 8,
    per_channel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    【对称绝对值最大量化 (Symmetric AbsMax)】
    特点: 零点 Zero Point 固定为 0，正负区间对称映射。
    计算公式:
      S = max(|x|) / q_max
      q = clamp(round(x / S), -q_max, q_max)
    
    :param x: 原始浮点张量 (FP32 或 FP16)
    :param bits: 目标位宽 (8 代表 INT8, 4 代表 INT4)
    :param per_channel: 是否沿输出维度 (dim=0) 逐通道量化 (大模型常用)
    :return: (量化后的整数张量 q, 缩放比例 Scale)
    """
    q_max = (2 ** (bits - 1)) - 1  # INT8 为 127; INT4 为 7
    q_min = -q_max                  # 对称截断: INT8 为 -127 (保留-128或对称使用); INT4 为 -7

    if per_channel:
        # 逐通道 (对权重矩阵沿 dim=0 计算每一行的最大值)
        # x 形状通常为 (out_features, in_features)
        max_val = torch.amax(torch.abs(x), dim=-1, keepdim=True)
    else:
        # 逐张量 (整个矩阵共用一个 Scale)
        max_val = torch.amax(torch.abs(x))

    # 避免除以 0
    scale = torch.clamp(max_val / q_max, min=1e-8)

    # 量化与舍入截断
    q = torch.clamp(torch.round(x / scale), q_min, q_max)

    # 根据位宽选用紧凑数据类型
    if bits <= 8:
        q = q.to(torch.int8)

    return q, scale


def dequantize_symmetric(
    q: torch.Tensor,
    scale: torch.Tensor,
    target_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    【对称反量化 (Dequantize)】
    将压缩后的低精度整数张量恢复为浮点张量。
    公式: x_hat = q * S
    """
    return q.to(target_dtype) * scale


def quantize_asymmetric(
    x: torch.Tensor,
    bits: int = 8,
    per_channel: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    【非对称最小-最大量化 (Asymmetric MinMax)】
    特点: 拥有非零的零点 Zero Point，能够完美对齐非对称分布 (如 ReLU/SiLU 激活值)。
    计算公式:
      S = (x_max - x_min) / (q_max - q_min)
      Z = round(-x_min / S) + q_min
      q = clamp(round(x / S) + Z, q_min, q_max)
    """
    q_min = 0
    q_max = (2 ** bits) - 1  # 8-bit 为 0~255; 4-bit 为 0~15

    if per_channel:
        x_min = torch.amin(x, dim=-1, keepdim=True)
        x_max = torch.amax(x, dim=-1, keepdim=True)
    else:
        x_min = torch.amin(x)
        x_max = torch.amax(x)

    scale = torch.clamp((x_max - x_min) / (q_max - q_min), min=1e-8)
    zero_point = torch.clamp(torch.round(-x_min / scale) + q_min, q_min, q_max)

    q = torch.clamp(torch.round(x / scale) + zero_point, q_min, q_max)

    if bits <= 8:
        q = q.to(torch.uint8)

    return q, scale, zero_point


def dequantize_asymmetric(
    q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    target_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    【非对称反量化】
    公式: x_hat = (q - Z) * S
    """
    return (q.to(target_dtype) - zero_point) * scale


def evaluate_quantization_error(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    """
    评估量化后与原始浮点数据之间的精度损失与误差指标
    """
    orig_f = original.to(torch.float32).flatten()
    recon_f = reconstructed.to(torch.float32).flatten()

    # 1. 余弦相似度 (Cosine Similarity, 衡量向量方向保真度, 越接近 1.0 越好)
    cos_sim = F.cosine_similarity(orig_f.unsqueeze(0), recon_f.unsqueeze(0)).item()

    # 2. 均方误差 (MSE, 均方差, 越小越好)
    mse = torch.mean((orig_f - recon_f) ** 2).item()

    # 3. 最大绝对误差 (Max Absolute Error)
    max_err = torch.max(torch.abs(orig_f - recon_f)).item()

    # 4. 信噪比 (SQNR, Signal-to-Quantization-Noise Ratio, 单位 dB, 越大越好)
    signal_power = torch.mean(orig_f ** 2).item()
    noise_power = mse
    sqnr_db = 10 * math.log10(signal_power / max(noise_power, 1e-12))

    return {
        "cosine_similarity": cos_sim,
        "mse": mse,
        "max_abs_error": max_err,
        "sqnr_db": sqnr_db
    }


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("================================================================")
    print("🔬 [实验 1: 微观展示] 观察 4x4 矩阵如何从浮点小数压缩为 INT8 整数")
    print("================================================================")

    torch.manual_seed(42)
    # 模拟真实大模型某一层的微型权重
    toy_weights = torch.randn(4, 4) * 1.5
    print("\n1. 原始权重 (FP32 小数):")
    print(toy_weights)

    # 执行对称 INT8 量化
    q_int8, scale_int8 = quantize_symmetric(toy_weights, bits=8, per_channel=False)
    print(f"\n2. 量化计算出的全局缩放因子 Scale: {scale_int8.item():.6f}")
    print("3. 压缩后的整数张量 (INT8 整数，每个数只占 1 字节):")
    print(q_int8)

    # 反量化重构
    reconstructed_weights = dequantize_symmetric(q_int8, scale_int8)
    print("\n4. 反量化恢复后的浮点权重 (Dequantized):")
    print(reconstructed_weights)

    metrics = evaluate_quantization_error(toy_weights, reconstructed_weights)
    print(f"\n[微观量化精度评估]:")
    print(f"- 余弦相似度: {metrics['cosine_similarity']:.6f} (极度逼近 1.0)")
    print(f"- 均方误差 (MSE): {metrics['mse']:.6e}")
    print(f"- 量化信噪比 (SQNR): {metrics['sqnr_db']:.2f} dB")

    print("\n================================================================")
    print("🔬 [实验 2: 粒度对比] 真实大尺寸权重 (1024x1024) 下 Per-Tensor vs Per-Channel")
    print("================================================================")
    
    # 构造含少许离群值 (Outliers) 的大型权重矩阵 (大模型真实权重分布特性)
    large_weights = torch.randn(1024, 1024)
    large_weights[0, 0] = 35.0  # 人工引入 1 个强离群值 (Outlier)

    # A. 逐张量量化 (Per-Tensor)
    q_tensor, s_tensor = quantize_symmetric(large_weights, bits=8, per_channel=False)
    rec_tensor = dequantize_symmetric(q_tensor, s_tensor)
    m_tensor = evaluate_quantization_error(large_weights, rec_tensor)

    # B. 逐通道量化 (Per-Channel / 每行独立 Scale)
    q_channel, s_channel = quantize_symmetric(large_weights, bits=8, per_channel=True)
    rec_channel = dequantize_symmetric(q_channel, s_channel)
    m_channel = evaluate_quantization_error(large_weights, rec_channel)

    print(f"| 量化粒度 | 余弦相似度 | 均方误差 (MSE) | 最大误差 | SQNR 信噪比 |")
    print(f"|---|---|---|---|---|")
    print(f"| **逐张量 (Per-Tensor)** | {m_tensor['cosine_similarity']:.6f} | {m_tensor['mse']:.6f} | {m_tensor['max_abs_error']:.4f} | {m_tensor['sqnr_db']:.2f} dB |")
    print(f"| **逐通道 (Per-Channel)** | {m_channel['cosine_similarity']:.6f} | {m_channel['mse']:.6f} | {m_channel['max_abs_error']:.4f} | {m_channel['sqnr_db']:.2f} dB |")
    print("\n💡 结论: 逐通道 (Per-Channel) 使得离群值只污染自身单行，其他所有行依然保持极高精度！")

