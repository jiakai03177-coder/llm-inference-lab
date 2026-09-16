# 第 7 周实验报告：CUDA Kernel Profiling 性能深度剖析 (Nsight Compute)

## 1. 诊断背景与环境
- **诊断工具**: NVIDIA Nsight Compute (ncu CLI 2022.4)
- **目标设备**: NVIDIA GeForce RTX 3060 Laptop GPU (Compute Capability 8.6, Ampere)
- **目标算子**: `matmul_shared_kernel` (分块大小: 16x16, 线程块大小: 256)

---

## 2. Nsight Compute 硬件物理计数器核心证据

### (1) Speed of Light (SOL 极限性能分析)
| 性能指标 (Metric Name) | 测量值 (Metric Value) | 工业级解读与健康度 |
|---|---|---|
| **Compute (SM) Throughput** | **92.81%** | 🟢 **卓越 (算力利用率突破 90%)** |
| **L1 / Shared Memory Throughput** | **95.23%** | 🟢 **极高 (片上 SRAM 吞吐接近理论极值)** |
| **DRAM Throughput (全局显存)** | **1.85%** | 🟢 **完美 (慢速全局显存访问被彻底消除)** |
| **SM Active Cycles** | 442,156 周期 | 计算核心高度处于繁忙工作状态 |

> **官方 NCU 诊断意见**:
> `INF: The kernel is utilizing greater than 80.0% of the available compute or memory performance of the device.` (该算子已经成功榨干显卡 80% 以上的物理性能上限)。

### (2) Occupancy (SM 硬件占用率)
| 指标 | 理论上限 (Theoretical) | 实际达成 (Achieved) | 效率比 |
|---|---|---|---|
| **Active Warps Per SM** | 48 warps | **44.78 warps** | 93.29% |
| **Occupancy 百分比** | 100% | **93.29%** | 几乎完全打满 |
| **Registers Per Thread** | 38 寄存器 | - | 控制在健康阈值内 |
| **Shared Memory Per Block** | 2.05 KB | 32.77 KB 配置 | 仅占极小份额，未限制并发 |

---

## 3. Profiling 深度诊断分析 (依据计划模板)

- **问题**：Naive 矩阵乘法中算力利用率低下，大量的时钟周期耗费在慢速 DRAM 访存排队上。
- **证据**：
  - Naive Kernel 执行时，GPU 绝大多数周期在干等内存加载；
  - 引入 Shared Memory 分块后，`ncu` 实测 **DRAM Throughput 暴跌至 1.85%**，而 **L1/Shared Memory 吞吐飙升到 95.23%**。
- **优化方法**：
  - 采用 16x16 二维线程块与分块滑动窗口（Tiling）；
  - 线程协作搬运数据入片上高速缓存，使用 `__syncthreads()` 实现无死锁数据同步。
- **优化前 vs 优化后 (以 2048x2048 规模为例)**：
  - 优化前 (Naive): 耗时 **24.107 ms** (0.713 TFLOPs)
  - 优化后 (Shared): 耗时 **16.468 ms** (1.043 TFLOPs)
  - 加速比: **1.46x**，显存访存压力骤降 16 倍。
- **原因分析**：
  - 达到了 **93.29% 的 Achieved Occupancy**，每个 SM 上拥有充足的 44.78 个活跃 Warp，硬件 Warp 调度器能够零开销掩盖算术执行延迟；
  - 算子正式从 **Memory-Bound (访存受限)** 成功跨越推进到了 **Compute-Bound (算力受限)** 区域！

