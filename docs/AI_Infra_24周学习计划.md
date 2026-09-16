# AI Infra：大模型推理基础设施 24 周学习计划

> 适用对象：具备 Python、PyTorch/CUDA 或 FPGA/HLS 基础，希望进入大模型推理基础设施（LLM Inference / AI Infra）方向的研究生或工程师。
>
> 建议投入：每周 12～15 小时。
>
> 最终目标：完成一个“基于 vLLM 的大模型推理服务性能分析与优化平台”，包含 KV Cache 实验、CUDA kernel 优化、量化部署、多 GPU 推理和完整 benchmark 报告。

---

## 一、总体学习目标

完成本计划后，应能够：

1. 解释 LLM 的 prefill、decode、KV Cache 和 batching。
2. 计算模型权重与 KV Cache 的基本显存需求。
3. 使用 CUDA 编写并优化简单 kernel。
4. 使用 Nsight Systems 和 Nsight Compute 定位性能瓶颈。
5. 阅读 vLLM 的 scheduler 和 KV Cache 相关代码。
6. 使用 vLLM、TensorRT-LLM 部署和压测模型。
7. 理解 INT8、INT4、FP8、FP4 等量化方法的差异。
8. 完成一个有数据、有图表、有源码的 AI Infra 项目。
9. 将 FPGA/HLS 与 LLM 推理加速结合，形成个人特色。

---

## 二、每周时间安排

| 内容 | 每周时间 | 目标 |
|---|---:|---|
| 理论学习 | 3 小时 | 理解原理和系统设计 |
| 代码实践 | 5～6 小时 | 实现或修改代码 |
| 性能实验 | 2～3 小时 | Benchmark 和 profiling |
| 总结输出 | 1～2 小时 | 技术笔记、图表、报告 |

建议平日每天学习 1～1.5 小时，周末安排一次 4～6 小时的集中实验。每周必须产生可运行代码或实验结果，避免连续数周只看课程。

---

# 第一阶段：推理基础（第 1～4 周）

## 第 1 周：环境与基线

### 学习内容

- Linux 常用命令。
- Python 虚拟环境。
- Git 和 GitHub。
- PyTorch 基本使用。
- GPU、CUDA、驱动、PyTorch CUDA 版本关系。
- `nvidia-smi` 基本用法。

### 实践任务

建立项目仓库：

```text
llm-inference-lab/
├── README.md
├── environment.yml
├── notebooks/
├── src/
├── benchmarks/
├── reports/
└── figures/
```

完成以下检查：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name())"
```

### 本周输出

- 环境配置文档。
- GPU 和 CUDA 信息记录。
- 一个 PyTorch GPU 矩阵乘法测试。

## 第 2 周：Transformer 推理流程

### 学习内容

- Tokenizer。
- Embedding。
- Self-Attention。
- MLP。
- Residual connection。
- RMSNorm/LayerNorm。
- Causal mask。
- Prefill 与 decode 的区别。

### 实践任务

用 PyTorch 实现一个简化 Transformer block，要求支持 causal mask，并测量不同 sequence length 下的耗时。

### 本周输出

- `simple_transformer.py`。
- sequence length 与 latency 图表。
- batch size 与 latency 图表。
- prefill 与 decode 差异说明。

## 第 3 周：KV Cache

### 学习内容

- 为什么 decode 阶段需要 KV Cache。
- 没有 KV Cache 时的重复计算。
- KV Cache 的张量形状。
- KV Cache 显存计算。
- MHA、MQA、GQA 的差别。

### 实践任务

实现两种推理方式：

1. 不使用 KV Cache。
2. 使用 KV Cache。

比较输出一致性、单 token decode 延迟、总生成时间和不同上下文长度下的显存变化。

### 本周输出

- `kv_cache_demo.py`。
- KV Cache 显存计算表。
- 性能对比图。

KV Cache 显存大致与下面因素成正比：

\[
\text{KV Cache Memory}
\propto
\text{layers}
\times
\text{sequence length}
\times
\text{batch size}
\times
\text{KV heads}
\times
\text{head dimension}
\times
\text{bytes per element}
\]

## 第 4 周：性能指标与 Benchmark

### 学习内容

- TTFT（Time To First Token）。
- TPOT（Time Per Output Token）。
- E2E latency。
- Throughput：tokens/s 或 requests/s。
- Goodput。
- P50、P95、P99 latency。
- GPU utilization、显存利用率、HBM bandwidth。
- Cost per million tokens。

### 实践任务

写一个 benchmark 工具：

```bash
python benchmark.py \
  --batch-size 4 \
  --input-len 512 \
  --output-len 128 \
  --dtype float16
```

记录输入长度、输出长度、batch size、TTFT、TPOT、总延迟、tokens/s 和显存占用。

### 阶段验收

能够回答：

- 为什么 batch size 增大后吞吐可能上升？
- 为什么延迟不一定下降？
- TTFT 和 TPOT 分别反映什么？
- 长上下文主要影响哪部分显存？

---

# 第二阶段：CUDA 与 GPU 性能优化（第 5～8 周）

## 第 5 周：CUDA 编程模型

### 学习内容

- Grid、Block、Thread。
- Warp。
- SIMT。
- Thread divergence。
- Global memory、Shared memory、Registers、L2 cache。
- Synchronization。

### 实践任务

实现并比较 CPU 向量加法、GPU 向量加法、不同 block size 和不同数据规模的性能。

### 本周输出

- CUDA C++ 源代码。
- block size 对性能影响图。
- 对 memory coalescing 的解释。

## 第 6 周：CUDA Memory 与矩阵乘法

### 学习内容

- Coalesced memory access。
- Shared memory tiling。
- Register blocking。
- Occupancy。
- Kernel launch overhead。
- Host-device transfer。

### 实践任务

实现三个版本的矩阵乘法：

1. Naive CUDA kernel。
2. Shared memory tiled kernel。
3. PyTorch/cuBLAS 对照版本。

比较 GFLOPS、kernel 时间、数据传输时间和不同矩阵规模下的差异。

## 第 7 周：CUDA Kernel Profiling

### 学习内容

- Kernel launch 参数。
- Occupancy。
- Warp execution efficiency。
- Memory throughput。
- Compute throughput。
- Branch efficiency。
- Roofline 基本思想。

### 实践任务

使用 Nsight Compute 分析矩阵乘法和简单 softmax kernel，找出性能瓶颈并重新优化。

### Profiling 报告模板

```text
问题：
证据：
优化方法：
优化前：
优化后：
原因分析：
```

## 第 8 周：Attention Kernel

### 学习内容

- Naive Attention。
- Tiling。
- Online softmax。
- IO-aware optimization。
- FlashAttention 基本思想。
- Prefill attention 与 decode attention 的差别。

### 实践任务

实现：

1. PyTorch reference attention。
2. 分块 attention。
3. 简化 CUDA attention kernel。

重点理解为什么减少 HBM 与 SRAM 之间的数据搬运能够提速，以及为什么避免显式保存完整 attention matrix。

### 阶段验收

能够解释：

- GPU kernel 慢的常见原因。
- 算力瓶颈与带宽瓶颈的区别。
- 为什么 FlashAttention 是 IO-aware algorithm。
- 为什么普通 CUDA kernel 不能简单等同于高性能 kernel。

---

# 第三阶段：vLLM 与推理服务（第 9～12 周）

## 第 9 周：vLLM 使用与 API 服务

### 学习内容

- 模型加载。
- OpenAI-compatible API。
- Streaming output。
- Sampling 参数。
- Batch inference。
- 基本服务监控。

### 实践任务

部署一个适合本机硬件的开源模型，完成单请求、多轮对话、流式输出、多并发请求和基础 API 测试。

### 本周输出

- 启动脚本。
- API 使用文档。
- 单请求和并发 benchmark。

## 第 10 周：Continuous Batching

### 学习内容

- Static batching。
- Dynamic batching。
- Continuous batching。
- Iteration-level scheduling。
- 请求加入和退出。
- 长短请求混合。

### 实践任务

构造三类 workload：

1. 输入短、输出短。
2. 输入长、输出短。
3. 输入短、输出长。

分别测试不同并发数和最大 token 数下的 TTFT、TPOT、吞吐和 P95 延迟。

## 第 11 周：PagedAttention 与 KV Cache 管理

### 学习内容

- 连续 KV Cache 的问题。
- Block-based KV Cache。
- Logical block 与 physical block。
- Block table。
- Block allocation。
- Block reuse。
- 显存碎片。

### 实践任务

实现简化版 block manager：

```python
class KVCacheBlockManager:
    def allocate(self, request_id, num_blocks):
        pass

    def append(self, request_id):
        pass

    def free(self, request_id):
        pass
```

支持新请求分配 block、生成过程中追加 block、请求完成后释放以及多请求管理。

## 第 12 周：阅读 vLLM 源码

### 建议顺序

1. 请求进入服务的路径。
2. Scheduler。
3. KV Cache 管理。
4. Worker。
5. ModelRunner。
6. Attention backend。
7. Sampling。

### 源码阅读问题

每阅读一个模块，回答：

- 输入是什么？
- 输出是什么？
- 它解决了什么性能或系统问题？

### 本周输出

完成“一个请求从 API 到 GPU 的完整路径”导读：

```text
API Server
→ Request
→ Scheduler
→ KV Cache Allocation
→ Model Execution
→ Sampling
→ Streaming Response
```

### 阶段验收

能够解释 PagedAttention 和 continuous batching 如何分别解决显存管理和动态调度问题。

---

# 第四阶段：量化与推理优化（第 13～16 周）

## 第 13 周：低精度基础

### 学习内容

- FP32、FP16、BF16。
- INT8、INT4、FP8、FP4。
- Dynamic range。
- Calibration。
- Quantization error。

### 实践任务

对一个小模型比较 FP16/BF16、INT8 和 INT4 推理，记录显存、TTFT、TPOT、吞吐和生成质量。

## 第 14 周：GPTQ、AWQ 与 SmoothQuant

### 学习内容

- Weight-only quantization。
- Activation quantization。
- GPTQ。
- AWQ。
- SmoothQuant。
- W4A16、W8A8。
- KV Cache quantization。

### 实践任务

建立量化对比表：

| 方法 | 权重精度 | 激活精度 | 显存 | 速度 | 精度 |
|---|---|---|---:|---:|---:|
| FP16 | FP16 | FP16 |  |  |  |
| INT8 | INT8 | FP16/INT8 |  |  |  |
| AWQ | INT4 | FP16 |  |  |  |
| GPTQ | INT4 | FP16 |  |  |  |

## 第 15 周：FlashAttention 与算子融合

### 学习内容

- Fused QKV。
- Fused RoPE。
- Fused RMSNorm。
- Fused MLP。
- FlashAttention。
- Decode attention。
- Chunked prefill。

### 实践任务

选择 RMSNorm、RoPE、Softmax、QKV projection 或简单 attention 中的一个，完成：

1. PyTorch baseline。
2. CUDA 或 Triton 版本。
3. 正确性测试。
4. 性能 benchmark。
5. Nsight profiling。

## 第 16 周：Speculative Decoding

### 学习内容

- Draft model。
- Target model。
- Draft tokens。
- Verification。
- Acceptance rate。
- 适用场景与限制。

### 实践任务

比较普通 decoding 和 speculative decoding，测试不同 draft token 数以及不同任务下的 acceptance rate。

### 阶段验收

能够回答：

- INT4 为什么能降低显存？
- 量化为什么可能损害精度？
- FlashAttention 主要减少了什么开销？
- Speculative decoding 什么时候不会提速？
- KV Cache quantization 有什么风险？

---

# 第五阶段：多 GPU 与 TensorRT-LLM（第 17～20 周）

## 第 17 周：多 GPU 基础

### 学习内容

- Data Parallelism。
- Tensor Parallelism。
- Pipeline Parallelism。
- Expert Parallelism。
- All-Reduce。
- All-Gather。
- Reduce-Scatter。
- NCCL。

### 实践任务

如果只有一张 GPU：

- 学习并运行 NCCL 示例。
- 使用多进程模拟通信。
- 阅读 Tensor Parallelism 的实现。
- 分析通信量。

如果有多张 GPU，则测试单 GPU、Tensor Parallel、不同 GPU 数量和通信开销。

## 第 18 周：GPU 互联与通信分析

### 学习内容

- PCIe。
- NVLink。
- GPU topology。
- P2P memory access。
- 通信与计算 overlap。
- InfiniBand/RDMA 基础。

### 实践任务

记录：

```bash
nvidia-smi topo -m
```

比较不同 GPU 对之间的通信性能，并计算多 GPU 扩展效率。

## 第 19 周：TensorRT-LLM

### 学习内容

- Model conversion。
- Engine build。
- Runtime。
- In-flight batching。
- Tensor parallelism。
- Quantization。
- KV Cache。
- Speculative decoding。

### 实践任务

对同一个模型进行 vLLM 与 TensorRT-LLM 部署和压测，保持模型、GPU、输入长度、输出长度、并发数和采样参数一致。

### 框架对比表

| 维度 | vLLM | TensorRT-LLM |
|---|---|---|
| 部署难度 |  |  |
| 模型适配 |  |  |
| TTFT |  |  |
| TPOT |  |  |
| 吞吐 |  |  |
| 显存 |  |  |
| 调试难度 |  |  |

## 第 20 周：Prefill-Decode 分离

### 学习内容

- Prefill-heavy workload。
- Decode-heavy workload。
- Disaggregated serving。
- KV Cache transfer。
- 网络传输成本。
- 独立扩容。

### 实践任务

设计如下架构：

```text
请求入口
   ↓
Router
   ├── Prefill Worker
   └── Decode Worker
```

画出系统架构，计算数据传输内容，并分析适用场景、收益和复杂度。

---

# 第六阶段：生产化项目与求职（第 21～24 周）

## 第 21 周：服务治理

### 学习内容

- Docker。
- FastAPI/gRPC。
- Prometheus。
- Grafana。
- 日志。
- Tracing。
- 限流、超时、重试和熔断。
- 健康检查。

### 实践任务

为推理服务增加：

- `/health`。
- `/metrics`。
- 请求 ID。
- 延迟统计。
- 错误日志。
- 并发限制。
- 最大输入长度限制。
- OOM 保护。

## 第 22 周：完整压测平台

### 实践任务

完成如下 benchmark pipeline：

```text
生成请求
  ↓
发送并发请求
  ↓
收集 token 时间戳
  ↓
计算 TTFT/TPOT/P95
  ↓
收集 GPU 指标
  ↓
生成 CSV 和图表
  ↓
输出 Markdown 报告
```

报告必须包括测试环境、测试模型、测试参数、性能结果、瓶颈判断、优化前后对比以及实验限制。

## 第 23 周：FPGA/HLS 特色项目

建议选择一个规模可控的方向：

### 选题 A：FPGA INT8 GEMM

- HLS 实现 INT8 矩阵乘法。
- 设计 tile/block。
- 分析片上缓存和带宽。
- 与 CUDA kernel 对比。

### 选题 B：FPGA LayerNorm/RMSNorm

- 实现 LayerNorm 或 RMSNorm。
- 比较流水线、并行度和资源消耗。
- 评估其作为 Transformer 推理子模块的可行性。

### 选题 C：GPU/FPGA 混合推理

- GPU 负责主要矩阵运算。
- FPGA 负责某个固定算子。
- 分析 PCIe 传输是否抵消加速收益。

### 选题 D：KV Cache 相关硬件加速

- 研究 block-based KV Cache。
- 分析 KV Cache 访问模式。
- 设计简化地址映射或缓存管理模块。

建议优先选择 A 或 B，因为更容易获得可测量结果。

## 第 24 周：简历与面试

### 简历项目写法

不要只写：

> 使用 vLLM 部署了一个大模型。

建议写成：

> 针对长短请求混合场景，基于 vLLM 搭建 LLM serving benchmark，测量 TTFT、TPOT、P95 latency、吞吐和显存占用；分析 continuous batching 与 KV Cache block allocation 对性能的影响，并完成量化部署和性能对比。

如果有真实数据，可以写成：

> 在固定输入长度、输出长度和并发条件下，将显存占用降低 X%，吞吐提升 Y%，P95 latency 降低 Z%。

X、Y、Z 必须来自实际实验，不能提前填写或估算。

### 高频面试问题

- Prefill 和 decode 的区别是什么？
- KV Cache 如何计算显存？
- 为什么 decode 受到显存带宽限制？
- PagedAttention 解决了什么问题？
- continuous batching 如何工作？
- FlashAttention 为什么更快？
- INT4、INT8、FP8 有什么区别？
- Tensor Parallelism 的通信成本是什么？
- 如何定位 GPU 利用率低？
- TTFT 很高但 TPOT 正常，可能是什么原因？
- P95 latency 很高而平均延迟正常，如何排查？
- 为什么量化后显存下降但速度没有提升？
- vLLM 与 TensorRT-LLM 应该如何选择？

---

# 三、推荐项目仓库结构

```text
llm-inference-lab/
├── README.md
├── environment.yml
├── docs/
│   ├── inference_basics.md
│   ├── kv_cache.md
│   ├── vllm_source_reading.md
│   └── profiling_report.md
├── src/
│   ├── transformer/
│   ├── kv_cache/
│   ├── cuda_kernels/
│   └── serving/
├── benchmarks/
│   ├── benchmark_pytorch.py
│   ├── benchmark_vllm.py
│   └── benchmark_trtllm.py
├── cuda/
│   ├── vector_add.cu
│   ├── matmul.cu
│   └── attention.cu
├── hls/
│   ├── int8_gemm/
│   └── rmsnorm/
├── reports/
└── figures/
```

---

# 四、每周固定工作模板

## 周一：理论

阅读一个主题，写出定义、解决的问题、核心机制、主要代价和一个具体例子。

## 周二至周四：实现

完成最小可运行代码：先保证正确，再做性能优化，最后加入异常处理和 benchmark。

## 周五：源码阅读

只阅读与当前主题相关的模块，并画调用流程图。

## 周末：实验与总结

完成至少一组对照实验、至少一张图、至少一个结论以及一篇 500～1000 字技术总结。

---

# 五、优先级与取舍

## 最优先学习

1. Prefill、Decode、KV Cache。
2. 显存占用计算与 GPU memory hierarchy。
3. PagedAttention 和 continuous batching。
4. CUDA kernel 与 Nsight profiling。
5. FlashAttention 和算子融合。
6. Quantization。
7. vLLM 源码。
8. Tensor Parallelism、NCCL 和多 GPU 部署。
9. TensorRT-LLM。
10. Kubernetes、监控和生产服务治理。

## 暂时不必优先

- 同时深入十几个 Agent 框架。
- 过早研究所有开源模型架构。
- 只背大模型面试八股。
- 一开始追逐最新模型名称。
- 没有 benchmark 就讨论哪个框架更快。
- 直接尝试从零训练超大模型。
- 在 FPGA 上直接部署完整 LLM。

---

# 六、最适合你的主线

```text
Transformer 推理原理
→ CUDA 性能优化
→ KV Cache 管理
→ vLLM 源码
→ 量化与推理部署
→ 多 GPU 通信
→ FPGA/HLS 异构加速
```

结合已有的 FPGA、HLS、CUDA 和 self-attention 背景，建议将求职定位逐步聚焦到：

> LLM Inference Engineer / AI Infra Engineer / GPU Kernel Optimization Engineer / AI Hardware-Software Co-design Engineer

最终项目建议命名为：

> **LLM Inference Lab：基于 vLLM、CUDA 与 FPGA/HLS 的大模型推理性能优化研究**

---

# 七、执行原则

- 每周至少提交一次代码或实验结果。
- 所有性能结论必须有测试条件和原始数据。
- 区分 correctness、latency、throughput、memory 和 quality。
- 先建立 baseline，再进行优化。
- 每次只改变一个主要变量。
- 记录失败实验，失败原因本身也是项目成果。
- 论文、源码和官方文档用于理解原理，最终以自己的实验结果为准。
