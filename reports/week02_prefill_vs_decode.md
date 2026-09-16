# 第 2 周实验报告：Transformer 推理流程与 Prefill/Decode 差异分析

## 1. 硬件配置
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU
- **模型规格**: Single Transformer Layer (Hidden Dim: 1024, Heads: 16, Dtype: FP16)

## 2. 测试数据汇总

### Prefill 阶段 (批量输入 Prompt)
| Sequence Length | 延迟 (ms) |
|---|---|
| 128 | 0.481 |
| 256 | 0.663 |
| 512 | 1.367 |
| 1024 | 3.011 |
| 2048 | 7.287 |

### Decode 阶段 (单 Token 自回归)
- **单步解码延迟 (S=1)**: **0.471 ms**

## 3. 核心现象与理论剖析
1. **Prefill 计算复杂度随长度迅速上升**：
   - 注意力机制计算 $Q K^T$ 的复杂度为 $O(S^2)$。随着 Prompt 长度翻倍，计算量呈平方级膨胀。
2. **Prefill 与 Decode 的本质鸿沟**：
   - Prefill 阶段（$S \ge 128$）矩阵尺寸大，能够喂饱 GPU 计算单元，属于 **Compute-bound（计算受限）**；
   - Decode 阶段每次只送入 1 个 token（$S=1$），矩阵退化成了向量乘矩阵（GEMV），算力跑不满，大量时间耗费在显存搬运权重上，属于 **Memory-bound（访存受限）**。
3. **引出下周核心问题——为什么必须要有 KV Cache？**
   - 如果在 Decode 阶段，我们每吐一个字，都把历史所有的上下文重新打包送进模型做一次完整的 Prefill，那么每一步生成都会是 $O(S^2)$ 的时间开销！这正是第 3 周要解决的问题。
