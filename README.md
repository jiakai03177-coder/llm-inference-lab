# 🚀 llm-inference-lab: 大模型推理基础设施 (AI Infra) 实战库

> 基于消费级显卡（**NVIDIA GeForce RTX 3060 Laptop 6GB**）从零构建的大模型推理优化体系。
> 涵盖 **CUDA 算子开发、FlashAttention、PagedAttention 分页管理、Continuous Batching、vLLM 服务化与低精度量化（INT8 / INT4 / AWQ）**。

---

## 📌 实验硬件与系统环境

* **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU (6GB GDDR6, GA106, 140W 满血版)
* **宿主环境**: Windows 11 + WSL2 (Ubuntu 24.04 LTS)
* **深度学习框架**: PyTorch 2.3+ / CUDA 12.x ~ 13.x
* **推理引擎**: vLLM 0.29.0+

---

## 🗺️ 24 周学习路线与里程碑进展

| 阶段 | 周次 | 核心主题 | 核心产出代码 / 脚本 | 技术实验报告 | 状态 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **第一阶段**<br>推理基础 | Week 01 | GPU 环境基线与 GEMM 矩阵乘法 | `benchmarks/baseline_gemm.py` | [week01_baseline.md](reports/week01_baseline.md) | ✅ 已完成 |
| | Week 02 | Transformer 架构与 Prefill / Decode 剖析 | `src/simple_transformer.py`<br>`benchmarks/benchmark_transformer.py` | [week02_prefill_vs_decode.md](reports/week02_prefill_vs_decode.md) | ✅ 已完成 |
| | Week 03 | KV Cache 机制原理与显存数学模型 | `benchmarks/kv_cache_demo.py` | [week03_kv_cache.md](reports/week03_kv_cache.md) | ✅ 已完成 |
| | Week 04 | 工业级推理 Benchmark 框架设计 | `benchmarks/run_benchmark.py` | [week04_benchmark.md](reports/week04_benchmark.md) | ✅ 已完成 |
| **第二阶段**<br>CUDA 算子加速 | Week 05 | CUDA 核心编程与向量加法并行化 | `src/vector_add.cu` | [week05_cuda_basics.md](reports/week05_cuda_basics.md) | ✅ 已完成 |
| | Week 06 | 片上共享内存 (Shared Memory) 分块 GEMM | `src/matmul_cuda.cu` | [week06_matmul.md](reports/week06_matmul.md) | ✅ 已完成 |
| | Week 07 | Nsight Compute / Systems 性能瓶颈剖析 | Profiling logs | [week07_profiling.md](reports/week07_profiling.md) | ✅ 已完成 |
| | Week 08 | FlashAttention 分块与 Online Softmax 算子 | `src/flash_attn_demo.cu` | [week08_attention_kernel.md](reports/week08_attention_kernel.md) | ✅ 已完成 |
| **第三阶段**<br>vLLM 与服务化 | Week 09 | vLLM 本地部署与 OpenAI 兼容流式 API | `scripts/chat_cli.py`<br>`benchmarks/vllm_offline_demo.py` | [week09_vllm_api.md](reports/week09_vllm_api.md) | ✅ 已完成 |
| | Week 10 | 连续批处理 (Continuous Batching) 动态调度压测 | `benchmarks/continuous_batching_benchmark.py` | [week10_continuous_batching.md](reports/week10_continuous_batching.md) | ✅ 已完成 |
| | Week 11 | 手写 PagedAttention 显存分页块管理器 | `src/block_manager.py`<br>`benchmarks/benchmark_block_manager.py` | [week11_paged_attention_block_manager.md](reports/week11_paged_attention_block_manager.md) | ✅ 已完成 |
| | Week 12 | 拆解阅读 vLLM 官方核心源码与端到端全链路 | 源码架构解析 | [week12_vllm_source_code_tour.md](reports/week12_vllm_source_code_tour.md) | ✅ 已完成 |
| **第四阶段**<br>量化与推理优化 | Week 13 | 低精度量化基础 (FP16 vs INT8 vs INT4) | `src/quantization_basics.py`<br>`benchmarks/benchmark_quantization.py` | [week13_quantization_basics.md](reports/week13_quantization_basics.md) | ✅ 已完成 |
| | Week 14 | 激活感知量化 (AWQ) 与 GPTQ 算法 | `src/awq_toy.py`<br>`benchmarks/benchmark_advanced_quant.py` | [week14_advanced_quantization.md](reports/week14_advanced_quantization.md) | ✅ 已完成 |

---

## 📂 仓库目录结构

```text
llm-inference-lab/
├── docs/                      # 学习大纲、配置指南与部署文档
│   ├── AI_Infra_24周学习计划.md
│   └── vllm_api_guide.md
├── src/                       # 核心算子与算法实现
│   ├── simple_transformer.py  # 现代 Transformer 架构 (RMSNorm + SwiGLU + Pre-Norm)
│   ├── block_manager.py       # PagedAttention 虚拟显存块管理器
│   ├── quantization_basics.py # 对称/非对称 INT8/INT4 量化与误差分析
│   ├── awq_toy.py             # 激活感知权重量化 (AWQ) 核心缩放保护实现
│   ├── vector_add.cu          # CUDA 基础向量加法核函数
│   ├── matmul_cuda.cu         # CUDA 共享内存分块矩阵乘法
│   └── flash_attn_demo.cu     # FlashAttention Online Softmax 简化核函数
├── benchmarks/                # 性能基准评测与仿真测试脚本
│   ├── baseline_gemm.py       # CPU vs GPU, FP32 vs FP16 TFLOPs 基准
│   ├── benchmark_transformer.py
│   ├── kv_cache_demo.py       # KV Cache 显存与单步延迟对比
│   ├── continuous_batching_benchmark.py
│   ├── benchmark_block_manager.py # 显存碎片率与并发仿真
│   ├── benchmark_quantization.py   # 4096 规模矩阵 4 种精度实测
│   └── benchmark_advanced_quant.py # FP16 vs INT4 vs AWQ vs GPTQ 核心算法大比拼
├── reports/                   # 历周实验分析报告与实测数据
├── scripts/                   # 交互工具与客户端
│   └── chat_cli.py            # 本地流式交互对话终端
├── .gitignore
└── README.md
```

---

## ⚡ 快速复现与运行测试

### 1. 运行 GEMM 算力基准测试 (测定显卡 TFLOPs 上限)
```bash
python benchmarks/baseline_gemm.py
```

### 2. 运行 PagedAttention 块管理器仿真 (对比显存碎片率与并发承载)
```bash
python benchmarks/benchmark_block_manager.py
```

### 3. 运行多精度量化基准实测 (FP32 vs FP16 vs INT8 vs INT4)
```bash
python benchmarks/benchmark_quantization.py
```

### 4. 运行极简 AWQ 保护量化演示 (对比普通 INT4 vs AWQ 保护)
```bash
python src/awq_toy.py
```

### 5. 运行进阶量化算法大比拼 (FP16 vs 普通 INT4 vs AWQ vs GPTQ)
```bash
python benchmarks/benchmark_advanced_quant.py
```


