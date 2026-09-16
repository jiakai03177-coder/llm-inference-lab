# 第 10 周技术报告：Continuous Batching 调度算法原理与三类工作负载工业级压测

- **实验周次**：Week 10（第三阶段：vLLM 与推理服务）
- **实验日期**：2026-09-15
- **测试硬件**：NVIDIA GeForce RTX 3060 Laptop GPU (6GB VRAM, GA106)
- **软件环境**：WSL2 (Ubuntu 24.04 LTS), CUDA Driver 616.92 (CUDA 13.4 UMD), vLLM 0.29.0
- **测试模型**：`Qwen/Qwen2.5-0.5B-Instruct` (FP16, 943 MB)
- **可视化产出**：[`figures/week10_continuous_batching.png`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/figures/week10_continuous_batching.png)

---

## 1. 核心理论：批处理架构的技术演进

在传统深度学习（如 CNN、BERT 等非自回归模型）中，批处理（Batching）是一个非常标准的过程：输入尺寸对齐，前向传播单次完成。然而在大语言模型（LLM）的**自回归解码（Autoregressive Generation）**场景下，传统的批处理机制暴露出致命缺陷。

```
传统静态批处理 (Static Batching) 的计算气泡 (Bubbles):
Req 1 [Prefill][D1][D2]------------------------> [PAD][PAD][PAD] (等待 Req 2 结束)
Req 2 [Prefill][D1][D2][D3]...[D100]...[D500] -> [结束]
                                                 ^ 严重的算力与显存气泡！

连续批处理 (Continuous Batching / Iteration-level Scheduling):
Iter 1: [Req 1 - D1]  [Req 2 - D1]
Iter 2: [Req 1 - D2]  [Req 2 - D2]
Iter 3: [Req 1 退出!] [Req 2 - D3] + [新到达 Req 3 立即插入 Prefill!]
        ^ 显存即刻释放, 算力无缝填充, 零气泡！
```

### 1.1 三代批处理机制对比

| 特性维度 | 静态批处理 (Static Batching) | 动态批处理 (Dynamic Batching) | 连续批处理 (Continuous Batching) |
| :--- | :--- | :--- | :--- |
| **代表框架** | 原始 PyTorch / TensorFlow | 早期的 Triton Inference Server | Orca (OSDI '22), vLLM (SOSP '23) |
| **调度粒度** | 请求级（Request-level） | 请求级（带时间窗口积攒） | **每次迭代级（Iteration-level）** |
| **退出机制** | 等待批次内最长请求结束 | 等待批次内最长请求结束 | **任一序列遇到 Stop Token 立即释放** |
| **插入机制** | 批次结束后才能载入新批次 | 窗口超时组包后统一开跑 | **下一步迭代新请求即插即算（Chunked Prefill）** |
| **显存浪费 (气泡)**| 极高（大量 `<pad>` 填充空转） | 极高（受最长序列木桶效应制约） | **极低（几乎完全消除时间气泡）** |
| **硬件利用率** | 随长尾请求激增而骤降 | 难以充分饱和 Tensor Core | **算力与访存带宽持续高度饱和** |

---

## 2. 实验设计：三类经典工业级工作负载 (Workloads)

我们使用 [`benchmarks/continuous_batching_benchmark.py`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/benchmarks/continuous_batching_benchmark.py) 脚本对 3 类极端场景进行压测，并发度阶梯递增：$C \in \{1, 2, 4, 8\}$。

1. **Workload 1: 短进短出 (Short In, Short Out)**
   - **典型场景**：意图识别、分类标注、简短 FAQ 问答。
   - **输入/输出规格**：Prompt ~25 tokens，Max Output ~30 tokens。
   - **系统特征**：生命周期极短，压力集中在请求调度吞吐率（QPS）。
2. **Workload 2: 长进短出 (Long In, Short Out)**
   - **典型场景**：长文档摘要、长代码审查、多文档 RAG 问答。
   - **输入/输出规格**：Prompt ~550 tokens（专业技术文档），Max Output ~30 tokens。
   - **系统特征**：**Prefill 算力受限型（Compute-bound）**，高度考验首 Token 延迟（TTFT）与 Prefix Caching 效率。
3. **Workload 3: 短进长出 (Short In, Long Out)**
   - **典型场景**：长故事创作、推理链（CoT）展开、完整代码工程生成。
   - **输入/输出规格**：Prompt ~25 tokens，Max Output ~220 tokens。
   - **系统特征**：**Decode 访存带宽受限型（Memory-bound）**，持续进行 KV Cache 自回归迭代。

---

## 3. 实测数据汇总与多维对比

以下数据由自动化压测套件在本地 RTX 3060 6GB 显卡上实测采集，毫秒级流式计时（数据持久化于 [`reports/week10_benchmark_data.json`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/reports/week10_benchmark_data.json)）：

### 3.1 核心性能数据汇总表

| 负载类型 | 并发度 (C) | 系统吞吐量 (tokens/s) | 平均 TTFT (ms) | 平均 TPOT (ms/token) | P95 端到端耗时 (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **短进短出** | 1 | 122.89 | 39.6 | 7.01 | 0.24 |
| (Short In, Short Out) | 2 | 232.84 | 56.7 | 6.79 | 0.26 |
| | 4 | 380.78 | 69.0 | 7.53 | 0.29 |
| | **8** | **650.39** | 124.4 | 6.66 | 0.32 |
| **长进短出** | 1 | 102.46 | 106.0 | 5.89 | 0.25 |
| (Long In, Short Out) | 2 | 183.34 | 120.3 | 6.97 | 0.32 |
| | 4 | 452.35 | 59.0* | 7.01 | 0.26 |
| | **8** | **864.46** | 58.5* | 6.86 | 0.26 |
| **短进长出** | 1 | 164.76 | 22.5 | 5.99 | 1.32 |
| (Short In, Long Out) | 2 | 302.02 | 34.8 | 6.48 | 1.45 |
| | 4 | 658.80 | 32.4 | 5.94 | 1.33 |
| | **8** | **1485.41** | 49.3 | **5.17** | **1.18** |

*(注: 带 `*` 号数据体现了 vLLM Prefix Caching 前缀缓存命中带来的 Prefill 延迟陡降效应)*

---

## 4. 关键性能曲线与现象剖析

图表详见：[`figures/week10_continuous_batching.png`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/figures/week10_continuous_batching.png)

```
                       三类负载性能表现全貌
 吞吐 (tokens/s)                                  首字延迟 TTFT (ms)
1500 +                       ▲ (1485.4)          140 +                  ● (124.4)
     |                     /                         |
1000 |                   /   ■ (864.5)           100 |  ■ (106.0)
     |                 /   /                         |
 500 |         ▲ (658.8) ● (650.4)                60 |         ■ (59.0)  ■ (58.5)
     |       /   /                                   |  ● (39.6)
   0 +---+---+---+---+                                0 +---+---+---+---+
         1   2   4   8 (并发数)                             1   2   4   8 (并发数)
     ▲: 短进长出  ■: 长进短出  ●: 短进短出
```

### 4.1 现象一：短进长出场景下，吞吐量呈现“爆发式”线性增长（高达 1485 tokens/s）
- **现象**：当并发数从 1 提升至 8 时，“短进长出”的系统总吞吐从 164.76 tokens/s 飙升到 **1485.41 tokens/s**（提升近 9 倍！）。
- **底层原理解析**：
  在单请求 Decode 阶段，GPU 必须为了生成 1 个 Token 而将整个 0.5B 模型（约 1GB 权重）从显存搬移到片上一次（算力/带宽比极低，严重受限于显存带宽）。
  而当 Continuous Batching 将 8 个长序列合并批处理时，**同一份模型权重从显存读入缓存后，同时服务于 8 个请求的向量计算**。显存搬移开销被平摊（Amortized），计算强度成倍上升，Tensor Core 得以全速运转，单字生成耗时（TPOT）甚至逆势下降到了 **5.17 ms/token**！

### 4.2 现象二：长进短出场景下，Prefix Caching 带来反直觉的 TTFT 锐减
- **现象**：在“长进短出”负载中，并发度 1 和 2 的 TTFT 超过 100ms，但在并发度 4 和 8 时，TTFT 反而暴跌并稳定在 **58.5 ms**。
- **底层原理解析**：
  在并发度为 4 和 8 时，多个客户端发送了带有相同文档上下文的请求。vLLM 的 **自动前缀缓存（Automatic Prefix Caching, APC）** 机制自动命中了公共 Prompt 的 KV Cache Block，直接跳过了耗时的 550 tokens 全量矩阵运算，使得后续请求的 Prefill 耗时接近于零！

### 4.3 现象三：P95 端到端延迟表现极其平稳
- **现象**：三类工作负载的 P95 延迟在并发数由 1 升至 8 的过程中几乎保持水平直线。
- **底层原理解析**：
  传统批处理会因为“最慢的请求”而拖慢整个批次的 P95 甚至 P99 延迟；而 Continuous Batching 是迭代级推进，短请求跑完当步立刻返回，长请求各自独立推进，互不阻塞，体现了极高的 QoS 服务质量与稳定性。

---

## 5. 本周交付清单

- [x] **Continuous Batching 压测套件**：[`benchmarks/continuous_batching_benchmark.py`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/benchmarks/continuous_batching_benchmark.py)
- [x] **四维性能可视化脚本**：[`benchmarks/plot_week10_results.py`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/benchmarks/plot_week10_results.py)
- [x] **高清性能对比分析图**：[`figures/week10_continuous_batching.png`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/figures/week10_continuous_batching.png)
- [x] **基准测试结构化原始数据**：[`reports/week10_benchmark_data.json`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/reports/week10_benchmark_data.json)
- [x] **第 10 周完整技术报告**：[`reports/week10_continuous_batching.md`](file:///c:/Users/asus/Desktop/LLMlearning/llm-inference-lab/reports/week10_continuous_batching.md)

