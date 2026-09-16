"""
第 9 周实践任务: vLLM OpenAI-Compatible API 客户端功能与性能验证脚本
涵盖:
1. 单请求测试 (Non-streaming completion)
2. 流式输出测试 (Streaming output / Server-Sent Events 打字机效果)
3. 多轮对话测试 (Multi-turn Chat)
4. 多并发压力基准测试 (Concurrent Benchmark, 计算吞吐与延迟)
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

API_BASE = "http://localhost:8000/v1"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def test_health():
    """验证服务健康状态"""
    print("\n" + "=" * 65)
    print("🔎 [测试 0] 检查 API 服务连通性与模型列表")
    print("=" * 65)
    try:
        resp = requests.get(f"{API_BASE}/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            print(f"✅ 服务运行正常! 可用模型: {[m['id'] for m in models]}")
            return True
        else:
            print(f"❌ 服务响应异常: HTTP {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到 {API_BASE}! 请确认 vLLM 服务端是否已成功启动。")
        return False

def test_single_chat():
    """测试 1: 单请求基础问答"""
    print("\n" + "=" * 65)
    print("💬 [测试 1] 单请求基础问答 (Non-streaming)")
    print("=" * 65)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "你是一名精通 AI 基础设施的专家。请用简洁通俗的语言回答。"},
            {"role": "user", "content": "什么是大模型推理中的 PagedAttention？它的主要作用是什么？"}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }

    start = time.perf_counter()
    resp = requests.post(f"{API_BASE}/chat/completions", json=payload, timeout=30)
    latency = time.perf_counter() - start

    if resp.status_code == 200:
        data = resp.json()
        answer = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        print(f"【问题】: {payload['messages'][1]['content']}")
        print(f"【回答】:\n{answer}")
        print("-" * 65)
        print(f"⏱️ 耗时: {latency:.2f}s | Prompt Tokens: {usage.get('prompt_tokens')} | Completion Tokens: {usage.get('completion_tokens')}")
    else:
        print(f"❌ 请求失败: {resp.text}")

def test_streaming_chat():
    """测试 2: 流式打字机输出 (Streaming Output)"""
    print("\n" + "=" * 65)
    print("🌊 [测试 2] 流式打字机输出测试 (Streaming Output)")
    print("=" * 65)

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "请写一段关于 GPU 显存带宽与算力平衡的思考："}
        ],
        "temperature": 0.7,
        "max_tokens": 160,
        "stream": True
    }

    print(f"【问题】: {payload['messages'][0]['content']}")
    print("【流式回答】: ", end="", flush=True)

    start = time.perf_counter()
    resp = requests.post(f"{API_BASE}/chat/completions", json=payload, stream=True, timeout=30)

    token_count = 0
    first_token_time = None

    for line in resp.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.perf_counter() - start
                        print(delta, end="", flush=True)
                        token_count += 1
                except Exception:
                    pass

    total_time = time.perf_counter() - start
    print("\n" + "-" * 65)
    ttft_str = f"{first_token_time:.3f}s" if first_token_time else "N/A"
    print(f"⏱️ 首 Token 延迟 (TTFT): {ttft_str} | 总耗时: {total_time:.2f}s | 流式输出: {token_count} chunks")

def test_multi_turn_chat():
    """测试 3: 多轮上下文记忆对话"""
    print("\n" + "=" * 65)
    print("🔄 [测试 3] 多轮对话与历史上下文记忆测试")
    print("=" * 65)

    messages = [
        {"role": "system", "content": "你是一位耐心的大模型架构师。"},
        {"role": "user", "content": "你好，我想在我的笔记本电脑（RTX 3060 6GB）上做大模型部署。"},
    ]

    print(f"【第一轮 - 用户】: {messages[1]['content']}")
    resp1 = requests.post(f"{API_BASE}/chat/completions", json={
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 80
    }).json()
    ans1 = resp1["choices"][0]["message"]["content"].strip()
    print(f"【第一轮 - 助手】: {ans1}\n")

    # 追加第一轮回答并提出追问 (测试上下文关联)
    messages.append({"role": "assistant", "content": ans1})
    messages.append({"role": "user", "content": "那我应该选择多大的模型比较合适？为什么？"})

    print(f"【第二轮 - 用户】: {messages[3]['content']}")
    resp2 = requests.post(f"{API_BASE}/chat/completions", json={
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 120
    }).json()
    ans2 = resp2["choices"][0]["message"]["content"].strip()
    print(f"【第二轮 - 助手】: {ans2}")

def _send_concurrent_req(req_id: int, prompt: str):
    """单个并发工作请求"""
    start = time.perf_counter()
    resp = requests.post(f"{API_BASE}/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64,
        "temperature": 0.5
    }, timeout=60)
    latency = time.perf_counter() - start
    tokens = 0
    if resp.status_code == 200:
        data = resp.json()
        tokens = data.get("usage", {}).get("completion_tokens", 0)
    return req_id, latency, tokens

def test_concurrency_benchmark(concurrency=8):
    """测试 4: 多并发压力与吞吐基准测试"""
    print("\n" + "=" * 65)
    print(f"⚡ [测试 4] 多并发压力与吞吐基准测试 (并发数 = {concurrency})")
    print("=" * 65)

    test_prompts = [
        "用一句话介绍 CUDA 的 Warp 机制：",
        "什么是连续批处理 (Continuous Batching)？",
        "为什么显存带宽是 LLM 推理的瓶颈？",
        "一句话解释 Online Softmax 在 FlashAttention 中的应用：",
        "PagedAttention 如何解决内存碎片？",
        "什么是 Prefill 阶段和 Decode 阶段？",
        "张量并行 (Tensor Parallelism) 的通信算子是什么？",
        "为正在调试 CUDA 算子的工程师写一句祝福："
    ]

    print(f"正在并发派发 {concurrency} 个推理任务...")
    overall_start = time.perf_counter()

    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(_send_concurrent_req, i + 1, test_prompts[i % len(test_prompts)])
            for i in range(concurrency)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    overall_time = time.perf_counter() - overall_start

    total_tokens = sum(r[2] for r in results)
    avg_latency = sum(r[1] for r in results) / len(results)
    throughput = total_tokens / overall_time if overall_time > 0 else 0

    print("\n📊 并发性能基准结果:")
    print(f"- 并发请求数     : {concurrency} 个")
    print(f"- 总生成 Token 数 : {total_tokens} tokens")
    print(f"- 整体耗时       : {overall_time:.2f} 秒")
    print(f"- 平均单请求延迟 : {avg_latency:.2f} 秒")
    print(f"- 系统端到端吞吐 : {throughput:.2f} tokens/s")
    print("=" * 65)

def main():
    if not test_health():
        return
    test_single_chat()
    test_streaming_chat()
    test_multi_turn_chat()
    test_concurrency_benchmark(concurrency=8)

if __name__ == "__main__":
    main()

