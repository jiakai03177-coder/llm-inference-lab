"""
第 10 周可视化脚本: 绘制 Continuous Batching 三类负载在不同并发下的关键性能曲线
生成图表: figures/week10_continuous_batching.png
包含 4 个核心维度:
1. 吞吐量曲线 (Throughput: tokens/s vs Concurrency)
2. 首 Token 延迟曲线 (TTFT: ms vs Concurrency)
3. 单元输出耗时曲线 (TPOT: ms/token vs Concurrency)
4. P95 端到端耗时对比 (P95 Latency: s vs Concurrency)
"""

import os
import json
import matplotlib.pyplot as plt

def plot_week10_metrics(data_path="reports/week10_benchmark_data.json", save_path="figures/week10_continuous_batching.png"):
    if not os.path.exists(data_path):
        print(f"❌ 数据文件不存在: {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    workloads = data.get("workloads", {})
    if not workloads:
        print("❌ 数据中未找到 workloads 记录")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 风格与配色配置
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    colors = {
        "short_in_short_out": "#2563eb",  # 科技蓝
        "long_in_short_out": "#16a34a",   # 翡翠绿
        "short_in_long_out": "#ea580c"    # 活力橙
    }
    labels_map = {
        "short_in_short_out": "Short In, Short Out",
        "long_in_short_out": "Long In, Short Out",
        "short_in_long_out": "Short In, Long Out"
    }
    markers = {
        "short_in_short_out": "o",
        "long_in_short_out": "s",
        "short_in_long_out": "^"
    }

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.suptitle("Continuous Batching Performance Benchmark (vLLM / Qwen2.5-0.5B / RTX 3060 6GB)", fontsize=16, fontweight="bold", y=0.98)

    # 1. 吞吐量对比 (Throughput)
    ax = axes[0, 0]
    for k, v in workloads.items():
        bms = v["benchmarks"]
        c_list = [b["concurrency"] for b in bms]
        th_list = [b["throughput"] for b in bms]
        label = labels_map.get(k, k)
        ax.plot(c_list, th_list, marker=markers.get(k, "o"), color=colors.get(k, "blue"), linewidth=2.2, markersize=8, label=label)
        for x, y in zip(c_list, th_list):
            ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9, fontweight="bold")
    ax.set_title("System Throughput vs Concurrency (Higher is Better)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Concurrency (Batch Size)", fontsize=11)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=11)
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    # 2. 首 Token 延迟 (TTFT)
    ax = axes[0, 1]
    for k, v in workloads.items():
        bms = v["benchmarks"]
        c_list = [b["concurrency"] for b in bms]
        ttft_list = [b["avg_ttft_ms"] for b in bms]
        label = labels_map.get(k, k)
        ax.plot(c_list, ttft_list, marker=markers.get(k, "o"), color=colors.get(k, "blue"), linewidth=2.2, markersize=8, label=label)
        for x, y in zip(c_list, ttft_list):
            ax.annotate(f"{y:.1f}ms", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    ax.set_title("Time-To-First-Token (TTFT) vs Concurrency (Lower is Better)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Concurrency (Batch Size)", fontsize=11)
    ax.set_ylabel("TTFT (milliseconds)", fontsize=11)
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    # 3. 单 Token 生成耗时 (TPOT)
    ax = axes[1, 0]
    for k, v in workloads.items():
        bms = v["benchmarks"]
        c_list = [b["concurrency"] for b in bms]
        tpot_list = [b["avg_tpot_ms"] for b in bms]
        label = labels_map.get(k, k)
        ax.plot(c_list, tpot_list, marker=markers.get(k, "o"), color=colors.get(k, "blue"), linewidth=2.2, markersize=8, label=label)
        for x, y in zip(c_list, tpot_list):
            ax.annotate(f"{y:.1f}ms", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    ax.set_title("Time-Per-Output-Token (TPOT) vs Concurrency (Lower is Better)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Concurrency (Batch Size)", fontsize=11)
    ax.set_ylabel("TPOT (ms/token)", fontsize=11)
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    # 4. P95 端到端延迟对比 (P95 Latency)
    ax = axes[1, 1]
    for k, v in workloads.items():
        bms = v["benchmarks"]
        c_list = [b["concurrency"] for b in bms]
        p95_list = [b["p95_latency_s"] for b in bms]
        label = labels_map.get(k, k)
        ax.plot(c_list, p95_list, marker=markers.get(k, "o"), color=colors.get(k, "blue"), linewidth=2.2, markersize=8, label=label)
        for x, y in zip(c_list, p95_list):
            ax.annotate(f"{y:.2f}s", (x, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9)
    ax.set_title("P95 End-to-End Latency vs Concurrency (Lower is Better)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Concurrency (Batch Size)", fontsize=11)
    ax.set_ylabel("P95 Latency (seconds)", fontsize=11)
    ax.set_xticks([1, 2, 4, 8])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="none")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"📊 性能对比分析图已成功生成: {save_path}")

if __name__ == "__main__":
    plot_week10_metrics()
