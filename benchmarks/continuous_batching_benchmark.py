"""
第 10 周实践任务: Continuous Batching (连续批处理 / 迭代级调度) 性能基准测试
评测目标:
1. 构造三类经典工业级工作负载 (Workload):
   - Workload 1: 短进短出 (Short-In, Short-Out: FAQ/意图识别)
   - Workload 2: 长进短出 (Long-In, Short-Out: 长文本摘要/文档问答)
   - Workload 3: 短进长出 (Short-In, Long-Out: 故事创作/长代码生成)
2. 在不同并发度 (Concurrency = 1, 2, 4, 8) 下精确采集:
   - TTFT (Time To First Token, 首字延迟)
   - TPOT (Time Per Output Token, 单字生成耗时)
   - 端到端单请求延迟 (Latency)
   - 系统端到端总吞吐 (Throughput: tokens/s)
   - P50, P95, P99 长尾分布
3. 输出结构化 JSON 数据供绘图和周报使用
"""

import os
import sys
import json
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# 构造长文本素材 (约 550 tokens) 用于 Workload 2 (长进短出)
LONG_CONTEXT_DOC = """
在大语言模型（LLM）的工程落地与基础设施建设中，推理性能的优化至关重要。自回归（Autoregressive）解码具有独特的两阶段执行特征：Prefill（预填充）阶段和 Decode（解码）阶段。
在 Prefill 阶段，模型并行接收用户输入的完整提示词（Prompt），计算注意力并将对应的 Key 和 Value 向量缓存到显存中（称为 KV Cache）。该阶段属于计算密集型（Compute-bound），高度依赖 GPU 的 Tensor Core 矩阵乘法算力。
而在 Decode 阶段，模型每生成一个新 Token，都需要将历史所有 Token 的 KV Cache 从显存高带宽读取到片上 SRAM 中进行注意力计算。每次仅生成单个 Token，属于典型的访存密集型（Memory-bandwidth bound），GPU 算力利用率通常极低。
为了解决传统静态批处理（Static Batching）导致的计算气泡（Bubbles）和资源浪费，Orca 与 vLLM 提出了连续批处理（Continuous Batching）机制。连续批处理将调度粒度细化至每次迭代（Iteration-level），允许不同序列在同一时刻分别处于不同的生成步长：已生成终止符的短请求在当前步结束后立即退出并释放显存；新到达的请求可在下一步立即插入执行 Prefill。
与此同时，结合 PagedAttention 虚拟内存分页管理技术，彻底消除了显存碎片，使得系统能够在有限的显存中容纳数倍的并发请求，大幅提升了数据中心的总体吞吐量（Throughput）并显著降低了单请求的端到端时延（Latency）。
"""

WORKLOAD_CONFIGS = {
    "short_in_short_out": {
        "name": "短进短出 (Short-In, Short-Out)",
        "desc": "意图识别/简要问答 (Prompt ~25 tokens, Output ~30 tokens)",
        "prompt": "请用一句话简要概括计算机网络中 TCP 三次握手的主要目的：",
        "max_tokens": 30
    },
    "long_in_short_out": {
        "name": "长进短出 (Long-In, Short-Out)",
        "desc": "长文本摘要/长文档问答 (Prompt ~550 tokens, Output ~30 tokens)",
        "prompt": f"请仔细阅读以下关于大模型推理基础设施的技术文章：\n{LONG_CONTEXT_DOC}\n\n问题：请用不超过两句话总结连续批处理（Continuous Batching）的核心价值：",
        "max_tokens": 30
    },
    "short_in_long_out": {
        "name": "短进长出 (Short-In, Long-Out)",
        "desc": "创意写作/复杂代码生成 (Prompt ~25 tokens, Output ~220 tokens)",
        "prompt": "请写一篇关于 GPU 显卡硬件架构与大模型推理显存瓶颈的深度思考短文，结构严谨，层次分明：",
        "max_tokens": 220
    }
}

CONCURRENCIES = [1, 2, 4, 8]

def send_streaming_request(req_id: int, prompt: str, max_tokens: int):
    """
    发送单个流式请求，毫秒级捕获 TTFT、总延迟与实际输出 token 数
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "max_tokens": max_tokens,
        "stream": True
    }

    start_time = time.perf_counter()
    first_token_time = None
    generated_tokens = 0

    try:
        resp = requests.post(API_URL, json=payload, stream=True, timeout=90)
        if resp.status_code != 200:
            print(f"❌ 请求 {req_id} 失败: HTTP {resp.status_code} - {resp.text}")
            return None

        for line in resp.iter_lines():
            if line:
                line_str = line.decode("utf-8", errors="ignore")
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            generated_tokens += 1
                    except Exception:
                        pass

        end_time = time.perf_counter()
        total_latency = end_time - start_time

        if first_token_time is None:
            # 未收到任何有效 token
            return None

        ttft = first_token_time - start_time
        # TPOT 计算: (总耗时 - 首字延迟) / (生成 token 数 - 1)
        decode_time = total_latency - ttft
        tpot = decode_time / (generated_tokens - 1) if generated_tokens > 1 else 0

        return {
            "req_id": req_id,
            "start_time": start_time,
            "end_time": end_time,
            "total_latency": total_latency,
            "ttft": ttft,
            "tpot": tpot,
            "generated_tokens": generated_tokens
        }

    except Exception as e:
        print(f"❌ 请求 {req_id} 异常: {e}")
        return None

def run_workload_benchmark(workload_key: str, workload_cfg: dict, concurrency: int):
    """
    针对指定工作负载与并发度执行压测批次
    """
    prompt = workload_cfg["prompt"]
    max_tokens = workload_cfg["max_tokens"]

    # 预热一次确保无 JIT 干扰
    send_streaming_request(0, "Warmup ping", 5)

    print(f"\n  ▶ 并发度 = {concurrency:2d} ... ", end="", flush=True)

    batch_start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_streaming_request, i + 1, prompt, max_tokens)
            for i in range(concurrency)
        ]
        for f in as_completed(futures):
            res = f.result()
            if res:
                results.append(res)

    batch_time = time.perf_counter() - batch_start

    if not results:
        print("❌ 全部请求失败")
        return None

    # 计算统计指标
    total_tokens = sum(r["generated_tokens"] for r in results)
    throughput = total_tokens / batch_time if batch_time > 0 else 0

    latencies = sorted([r["total_latency"] for r in results])
    ttfts = sorted([r["ttft"] for r in results])
    tpots = [r["tpot"] for r in results if r["tpot"] > 0]

    def percentile(data, p):
        idx = int(len(data) * p)
        return data[min(idx, len(data) - 1)]

    avg_ttft = statistics.mean(ttfts)
    p95_ttft = percentile(ttfts, 0.95)

    avg_tpot = statistics.mean(tpots) if tpots else 0
    avg_latency = statistics.mean(latencies)
    p95_latency = percentile(latencies, 0.95)

    print(f"完成! [吞吐: {throughput:6.2f} tok/s | 均均 TTFT: {avg_ttft*1000:6.1f}ms | 平均 TPOT: {avg_tpot*1000:5.2f}ms/tok | P95耗时: {p95_latency:5.2f}s]")

    return {
        "concurrency": concurrency,
        "total_requests": len(results),
        "total_tokens": total_tokens,
        "batch_elapsed_time": batch_time,
        "throughput": throughput,
        "avg_ttft_ms": avg_ttft * 1000,
        "p95_ttft_ms": p95_ttft * 1000,
        "avg_tpot_ms": avg_tpot * 1000,
        "avg_latency_s": avg_latency,
        "p95_latency_s": p95_latency,
        "details": results
    }

def main():
    print("=" * 70)
    print("🚀 第 10 周: Continuous Batching (连续批处理) 工业级三类负载压测")
    print(f"目标服务: {API_URL} | 模型: {MODEL_NAME}")
    print("=" * 70)

    # 检查服务存活性
    try:
        chk = requests.get(f"http://localhost:8000/v1/models", timeout=5)
        if chk.status_code != 200:
            print("❌ vLLM API 服务未就绪，请先启动 scripts/start_vllm_server.sh！")
            return
    except Exception:
        print("❌ 无法连接到 http://localhost:8000，请先启动 scripts/start_vllm_server.sh！")
        return

    all_data = {
        "model": MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "workloads": {}
    }

    for wk_key, wk_cfg in WORKLOAD_CONFIGS.items():
        print(f"\n======================================================================")
        print(f"📂 正在评估负载类型: {wk_cfg['name']}")
        print(f"   说明: {wk_cfg['desc']}")
        print(f"======================================================================")

        wk_results = []
        for c in CONCURRENCIES:
            res = run_workload_benchmark(wk_key, wk_cfg, c)
            if res:
                # 剥离原始 request 细项以保持 JSON 紧凑
                summary_res = {k: v for k, v in res.items() if k != "details"}
                wk_results.append(summary_res)

        all_data["workloads"][wk_key] = {
            "name": wk_cfg["name"],
            "desc": wk_cfg["desc"],
            "benchmarks": wk_results
        }

    # 保存基准测试数据
    os.makedirs("reports", exist_ok=True)
    out_file = "reports/week10_benchmark_data.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"✅ 三类负载压测全流程执行完毕！数据已保存至: {out_file}")
    print("=" * 70)

if __name__ == "__main__":
    main()

