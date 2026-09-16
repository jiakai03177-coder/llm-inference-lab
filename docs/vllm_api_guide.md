# vLLM OpenAI-Compatible API 部署与使用指南 (Week 9)

本项目提供生产级、完全兼容 OpenAI 标准协议的大语言模型推理服务部署方案，基于 **vLLM 0.29.0** 与 **Qwen2.5-0.5B-Instruct** 模型，针对消费级显卡（NVIDIA GeForce RTX 3060 Laptop 6GB VRAM）及 WSL2 环境进行了深度适配与显存保护。

---

## 1. 架构与设计原则

```
+-------------------------------------------------------------------+
|                        Client Applications                        |
|  (curl / OpenAI Python SDK / Web UI / LangChain / Agent CLI)      |
+---------------------------------+---------------------------------+
                                  | HTTP REST / SSE (Port 8000)
+---------------------------------v---------------------------------+
|                       vLLM API Server (FastAPI)                   |
|  - Endpoint: /v1/chat/completions (POST)                          |
|  - Endpoint: /v1/models (GET)                                     |
+---------------------------------+---------------------------------+
                                  | Async Engine Core
+---------------------------------v---------------------------------+
|                      vLLM Execution Engine                        |
|  - Continuous Batching (动态连续批处理调度器)                       |
|  - PagedAttention (分块分页 KV Cache 虚拟显存管理)                 |
|  - CUDA Graph Optimization (减少小批次 CPU Launch 开销)           |
+---------------------------------+---------------------------------+
                                  | CUDA 13.4 UMD
+---------------------------------v---------------------------------+
|               NVIDIA GeForce RTX 3060 Laptop (6GB VRAM)           |
+-------------------------------------------------------------------+
```

---

## 2. 服务启动与配置

### 2.1 启动命令

在已激活 `vllm-env` 环境的 WSL2 终端中运行：

```bash
bash scripts/start_vllm_server.sh
```

### 2.2 核心配置参数解析

| 参数名 | 推荐取值 | 说明 |
| :--- | :--- | :--- |
| `--model` | 本地快照绝对路径 | 彻底离线加载，绕过 Hub 联网超时 |
| `--served-model-name` | `Qwen/Qwen2.5-0.5B-Instruct` | 对外暴露的模型 ID，客户端据此路由 |
| `--host` | `0.0.0.0` | 允许宿主机 Windows 通过 `localhost:8000` 跨系统直接访问 |
| `--port` | `8000` | HTTP 服务监听端口 |
| `--gpu-memory-utilization`| `0.75` | 限制显存占用不超过 75%（约 4.5GB），防止 6GB 显卡 OOM |
| `--max-model-len` | `2048` | 最大上下文长度，平稳分配 PagedAttention 显存池 |

---

## 3. API 接口调用示例

### 3.1 检查服务健康状态与可用模型
```bash
curl http://localhost:8000/v1/models
```

### 3.2 单次非流式补全 (Chat Completion)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "system", "content": "你是一位优秀的 AI 基础设施工程师。"},
      {"role": "user", "content": "请用一句话解释 Continuous Batching 的好处："}
    ],
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

### 3.3 流式输出 (Streaming Output / SSE 打字机)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "请列举 GPU 显存带宽优化的三个关键点："}
    ],
    "stream": true,
    "max_tokens": 128
  }'
```

---

## 4. Python SDK 快速集成

由于接口与官方 OpenAI 完全兼容，可直接使用标准 `openai` 客户端库无缝接入：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"  # vLLM 本地服务无需鉴权
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    messages=[
        {"role": "user", "content": "你好！请自我介绍一下。"}
    ],
    stream=True
)

for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print()
```

---

## 5. 自动化测试与压测套件

运行集成测试脚本：
```bash
python benchmarks/test_vllm_api.py
```
该脚本将依次自动化执行：
1. 服务连通性验证
2. 单请求延迟测试
3. 流式打字机响应与首 Token 延迟（TTFT）统计
4. 多轮对话上下文延续性验证
5. 8 线程并发压力测试与吞吐量（tokens/s）统计

