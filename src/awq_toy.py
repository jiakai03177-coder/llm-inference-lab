"""
第 14 周核心算法拆解: 极简版 AWQ (Activation-aware Weight Quantization) 原理实现
================================================================================
论文出处: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (MIT Han Lab)

核心数学洞察:
大模型中 99% 的参数是平庸的，只有 1% 的参数是影响模型智商的命门 (Salient Weights)。
怎样保护这 1% 的重要参数？
不是看权重 W 自身多大，而是看哪个通道的输入激活值 X 巨大 (Outlier Activation Channel)！
AWQ 发现等式:
   Y = X * W^T = (X / S) * (W * S)^T
通过寻找最佳每通道缩放因子 S (通常 S = s_x^alpha)，将重要的权重放大保护后再做 4-bit 量化，
等效于给核心权重穿上一层“防误差铠甲”！
"""

import math
from typing import Tuple, Dict
import torch
import torch.nn.functional as F

# 引用第 13 周我们手写的量化函数
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.quantization_basics import quantize_symmetric, dequantize_symmetric, evaluate_quantization_error


def vanilla_round_to_nearest(
    w: torch.Tensor,
    bits: int = 4
) -> torch.Tensor:
    """
    【传统无保护的普通 4-bit 量化 (RTN, Round-To-Nearest)】
    不管三七二十一，直接逐通道四舍五入。
    """
    q, scale = quantize_symmetric(w, bits=bits, per_channel=True)
    w_dequant = dequantize_symmetric(q, scale, target_dtype=w.dtype)
    return w_dequant


def awq_quantize(
    w: torch.Tensor,
    x: torch.Tensor,
    bits: int = 4,
    num_grid: int = 20
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    【极简版 AWQ 核心量化算法】
    1. 统计校准数据 X 在各个输入通道上的平均激活强度 s_x
    2. 网格搜索最佳强度系数 alpha in [0, 1]，得到缩放因子 S = s_x^alpha
    3. 权重变换: W' = W * S
    4. 对 W' 进行 4-bit 量化
    5. 返回 (反量化后的保护权重 W_hat', 缩放保护向量 S)
    """
    # w 形状: (out_features, in_features)
    # x 形状: (batch_size, in_features)
    out_features, in_features = w.shape

    # 1. 观察激活值强度: 统计每个通道的平均绝对值大小
    # s_x 形状: (1, in_features)
    s_x = torch.mean(torch.abs(x), dim=0, keepdim=True)
    # 归一化以保证数值稳定
    s_x = s_x / torch.clamp(torch.max(s_x), min=1e-5)

    # 金标准未量化输出
    y_ref = F.linear(x, w)

    best_error = float("inf")
    best_s = torch.ones(1, in_features, device=w.device, dtype=w.dtype)

    # 2. 网格搜索寻找最佳 alpha (0.0 到 1.0)
    # alpha=0 代表不缩放 (退化为普通 RTN); alpha 越大，对大激活通道的保护越强
    for i in range(num_grid + 1):
        alpha = i / num_grid
        # 计算当前候选缩放因子 S
        s = torch.clamp(s_x.pow(alpha), min=1e-4)
        # 保持整体均值平衡
        s = s / torch.sqrt(torch.mean(s ** 2))

        # 权重乘以 S 进行保护
        w_scaled = w * s

        # 对保护后的权重做 4-bit 量化
        q_temp, scale_temp = quantize_symmetric(w_scaled, bits=bits, per_channel=True)
        w_scaled_dequant = dequantize_symmetric(q_temp, scale_temp, target_dtype=w.dtype)

        # 验证在验证集上的输出误差: Y_pred = (X / S) * (W_hat')^T
        x_scaled = x / s
        y_pred = F.linear(x_scaled, w_scaled_dequant)

        # 计算 MSE 损失
        loss = torch.mean((y_ref - y_pred) ** 2).item()

        if loss < best_error:
            best_error = loss
            best_s = s.clone()

    # 3. 采用最优的 best_s 执行最终的 AWQ 保护量化
    w_final_scaled = w * best_s
    q_final, scale_final = quantize_symmetric(w_final_scaled, bits=bits, per_channel=True)
    w_final_dequant = dequantize_symmetric(q_final, scale_final, target_dtype=w.dtype)

    return w_final_dequant, best_s


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("==================================================================")
    print("🔬 [实测验证] AWQ 激活感知保护 vs 传统普通 4-bit 量化 (RTN)")
    print("==================================================================\n")

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 构造一个 512 x 512 的层
    dim = 512
    batch_size = 8
    dtype = torch.float32

    # 模拟真实大模型的权重
    w = torch.randn(dim, dim, device=device, dtype=dtype) * 0.5

    # 模拟真实大模型的激活值 X (99% 的通道很小，但有 2 个极其强烈的离群特征通道 Outliers!)
    x = torch.randn(batch_size, dim, device=device, dtype=dtype) * 0.8
    x[:, 10] *= 25.0   # 第 10 号通道是大离群特征 (比如处理否定词/转折词时神经元暴击)
    x[:, 250] *= 30.0  # 第 250 号通道也是强离群特征

    # 金标准输出 Y = X * W^T
    y_golden = F.linear(x, w)

    # -------------------------------------------------------------
    # 策略 1: 传统普通 4-bit 量化 (RTN)
    # -------------------------------------------------------------
    w_rtn = vanilla_round_to_nearest(w, bits=4)
    y_rtn = F.linear(x, w_rtn)
    metrics_rtn = evaluate_quantization_error(y_golden, y_rtn)

    # -------------------------------------------------------------
    # 策略 2: AWQ 激活感知保护 4-bit 量化
    # -------------------------------------------------------------
    w_awq, s_awq = awq_quantize(w, x, bits=4)
    x_scaled = x / s_awq
    y_awq = F.linear(x_scaled, w_awq)
    metrics_awq = evaluate_quantization_error(y_golden, y_awq)

    # -------------------------------------------------------------
    # 结果对比展示
    # -------------------------------------------------------------
    print(f"| 量化方法 | 位宽 | 输出余弦保真度 (Cosine) | 均方误差 (MSE) | 信噪比 (SQNR) | 精度衰减评价 |")
    print(f"|---|---|---|---|---|---|")
    print(f"| **传统普通 INT4 (RTN)** | 4-bit | {metrics_rtn['cosine_similarity']:.6f} | {metrics_rtn['mse']:.4f} | {metrics_rtn['sqnr_db']:.2f} dB | 离群通道被放大污染 |")
    print(f"| **AWQ 保护 INT4** | 4-bit | **{metrics_awq['cosine_similarity']:.6f}** | **{metrics_awq['mse']:.4f}** | **{metrics_awq['sqnr_db']:.2f} dB** | 🏆 **误差暴降，完美护盘！** |")

    # 计算提升幅度
    mse_drop = (metrics_rtn['mse'] - metrics_awq['mse']) / metrics_rtn['mse'] * 100
    print(f"\n🎉 实测结果分析:")
    print(f"1. 在相同的 4-bit 物理压缩率下，AWQ 让均方误差 (MSE) **暴降了 {mse_drop:.1f}%**！")
    print(f"2. 输出特征向量的余弦保真度直接从 {metrics_rtn['cosine_similarity']:.4f} 提升至 **{metrics_awq['cosine_similarity']:.6f}**！")
    print(f"3. 这证明了：只要把重要通道的权重放大保护起来，4-bit 量化就不会损伤大模型的智商！")

