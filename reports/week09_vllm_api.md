# 第 9 周技术报告：vLLM 推理引擎部署、OpenAI 兼容 API 与高并发基准测试

- **实验周次**：Week 9（第三阶段：vLLM 与推理服务）
- **实验日期**：2026-09-15
- **测试硬件**：NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM, GA106)
- **软件环境**：WSL2 (Ubuntu 24.04 LTS), CUDA Driver 616.92 (CUDA 13.4 UMD), PyTorch 2.13.0+cu126, vLLM 0.29.0
- **测试模型**：`Qwen/Qwen2.5-0.5B-Instruct` (FP16 / Safetensors, 943 MB)

---

## 1. 实验目标与背景

前 8 周我们深入了底层算子演进：从原生 PyTorch 到手写 CUDA Vector Add、从 Tiled Shared Memory GEMM 矩阵乘到 Nsight Compute 性能剖析，再到手写 FlashAttention Online Softmax 核函数。

进入第 9 周，我们的目标是搭建工业级大模型推理基础设施（LLM Inference Serving）：
1. 验证 **vLLM** 离线批量推理（Offline Batch Inference）能力。
2. 部署 **OpenAI-Compatible API Server**，使本地大模型具备生产级 Web 服务能力。
3. 评估单请求非流式、流式打字机输出（Streaming / SSE）、多轮对话上下文以及多并发请求（8 并发）的端到端吞吐量与首 Token 延迟（TTFT）。

---

## 2. 关键技术难点与解决方案

在将 6GB 显存显卡与 WSL2 环境接入 vLLM 0.29.0（基于 CUDA 13 编译）的过程中，我们攻克了以下几个关键工程瓶颈：

1. **宿主驱动与 UMD 匹配**：
   - 现象：旧驱动 566.07 仅支持至 CUDA 12.7，导致动态库报 `The NVIDIA driver on your system is too old (12070)`。
   - 方案：升级 Windows NVIDIA 驱动至 **616.92**，使 WSL2 UMD 升级至 **CUDA 13.4**，完全向前兼容 vLLM 的 CUDA 13.0 预编译算子。
2. **WSL2 UVA 与统一内存限制**：
   - 现象：WSL2 虚拟化环境下统一虚拟地址空间（UVA）默认不可用，报 `RuntimeError: UVA is not available`。
   - 方案：开启 `export VLLM_WSL2_ENABLE_PIN_MEMORY=1`，确保内存锁页分配在 WSL2 虚拟化总线中正常流转。
3. **FlashInfer JIT 编译依赖规避**：
   - 现象：FlashInfer 的 JIT 采样器使用了高版本 nvcc 编译选项 `--compress-mode=size`。
   - 方案：配置 `export VLLM_USE_FLASHINFER_SAMPLER=0`，平滑回退至 PyTorch 原生高效采样器。
4. **离线隔离与网络超时规避**：
   - 现象：国内网络环境下，Hugging Face Hub 与 vLLM 遥测模块持续发起海外请求，导致 TCP SYN 阻塞超时（耗时达数分钟）。
   - 方案：配置 `HF_HUB_OFFLINE=1`、`VLLM_NO_USAGE_STATS=1`，并将模型路径锁定为本地快照绝对路径，达成**零网络延迟、秒级载入**。

---

## 3. 基准测试数据与实验结果

### 3.1 离线批量推理基准测试（`benchmarks/vllm_offline_demo.py`）

- **测试配置**：Batch Size = 4 并发提示词，`gpu_memory_utilization = 0.75`，`max_model_len = 2048`。
- **实测指标**：
  - 总生成 Token 数：**512 tokens** (4 × 128)
  - 端到端耗时：**1.02 秒**
  - **系统吞吐量 (Throughput)**：**503.93 tokens/s**

> **分析**：得益于 Continuous Batching 与 PagedAttention，4 个序列在 Prefill 阶段合并成统一 Tensor 进行矩阵运算，Decode 阶段并行推进，在 RTX 3060 上跑出了超过 500 tokens/s 的极高吞吐。

---

### 3.2 在线 API Server 综合测试（`benchmarks/test_vllm_api.py`）

服务端启动参数：
```bash
vllm serve $MODEL_PATH --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.75 --max-model-len 2048
```

| 测试场景 | 请求类型 | 输入/输出规模 | 首 Token 延迟 (TTFT) | 端到端耗时 | 吞吐表现 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **单请求基础测试** | 非流式 POST | 46 prompt / 150 completion | - | 0.95 s | ~158 tokens/s |
| **流式打字机测试** | SSE 流式输出 | 28 prompt / 160 completion | **35 ms (0.035s)** | 0.78 s | ~205 tokens/s |
| **多轮对话上下文** | 2 轮连续对话 | 累计 180+ tokens | - | 1.12 s | 上下文逻辑严密、无记忆丢失 |
| **8 线程高并发压测** | 8 客户端并发 | 8 请求共 458 completion | 平均 42 ms | 2.96 s | **154.54 tokens/s** |

---

## 4. 关键结论与性能分析

1. **超低首字延迟（TTFT = 35ms）**：
   在流式测试中，客户端在发出请求后仅 35 毫秒便接收到了第一个 Token 的字节流推送。这归功于 0.5B 小模型的轻量 Prefill 阶段与 CUDA Graph 捕获消除的 Kernel 启动开销。
2. **并发弹性（Continuous Batching 的实际威力）**：
   在 8 个客户端几乎同一瞬间并发请求时，vLLM 自动将 8 个请求纳入同一迭代调度器中：
   - 整体耗时仅 2.96 秒就完成了全部 8 个长难任务的生成（458 个 Token）；
   - 平均每个客户端感知到的耗时为 2.92 秒；
   - 系统整体吞吐保持在 **154.54 tokens/s**，显存稳占 4.48 GB（在 6GB 安全红线以内，未发生任何 OOM）。
3. **OpenAI 兼容性无缝衔接**：
   对外暴露的 `/v1/chat/completions` 与 `/v1/models` 完全支持业界标准协议，能够无缝作为各类前端 WebUI、LangChain、AutoGPT 等上层系统的本地后端。

---

## 5. 本周产出清单

- [x] **服务启动脚本**：[`scripts/start_vllm_server.sh`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/scripts/start_vllm_server.sh)
- [x] **离线吞吐测试脚本**：[`benchmarks/vllm_offline_demo.py`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/benchmarks/vllm_offline_demo.py)
- [x] **API 自动化测试套件**：[`benchmarks/test_vllm_api.py`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/benchmarks/test_vllm_api.py)
- [x] **API 使用与对接文档**：[`docs/vllm_api_guide.md`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/docs/vllm_api_guide.md)
- [x] **第 9 周完整技术报告**：[`reports/week09_vllm_api.md`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/reports/week09_vllm_api.md)

