import os
# 关键设置：自动使用国内 HuggingFace 镜像源，防止海外网络超时
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
cu13_path = "/home/asus/vllm-env/lib/python3.12/site-packages/nvidia/cu13/lib"
if os.path.exists(cu13_path):
    os.environ["LD_LIBRARY_PATH"] = cu13_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

import time
from vllm import LLM, SamplingParams

def main():
    print("=" * 60)
    print("🚀 第 9 周实验: vLLM 离线批量推理与性能基准测试")
    print("=" * 60)

    # 1. 配置模型
    # Qwen2.5-0.5B-Instruct 只有约 1GB 显存，极度适合 6GB 显卡高速运行
    local_snapshot = "/home/asus/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
    model_name = local_snapshot if os.path.exists(local_snapshot) else "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"\n[1/3] 正在加载模型: {model_name}...")
    # gpu_memory_utilization=0.75: 限制 vLLM 最多使用 75% 的显存 (留给系统和防 OOM)
    # max_model_len=2048: 设置最大上下文长度
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.75,
        max_model_len=2048,
        trust_remote_code=True
    )

    # 2. 配置生成采样参数 (SamplingParams)
    # temperature=0.7: 采样随机度 (0 为贪婪匹配，越大回答越多样)
    # top_p=0.8: 核采样阈值
    # max_tokens=128: 限制每次回答最多生成的 token 数量
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        max_tokens=128
    )

    # 3. 准备并发提示词 (Batch Prompts)
    prompts = [
        "请用一句话解释什么是大模型推理中的 KV Cache：",
        "为什么 GPU 适合进行矩阵并行计算？简要说明：",
        "什么是 FlashAttention 算法的核心思想？一句话概括：",
        "请为刚入坑 AI Infra 的同学写一句鼓励的话："
    ]

    print(f"\n[2/3] 提交批量请求 (Batch Size = {len(prompts)})...")
    
    # 4. 执行批量生成并计时
    start_time = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.perf_counter() - start_time

    # 5. 打印生成结果与吞吐统计
    print("\n[3/3] 模型生成结果输出展示:\n" + "-" * 60)
    total_tokens_generated = 0

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        tokens_len = len(output.outputs[0].token_ids)
        total_tokens_generated += tokens_len

        print(f"【请求 {i+1}】: {prompt}")
        print(f"【回答】: {generated_text.strip()}")
        print(f"【生成 Token 数】: {tokens_len}\n")

    throughput = total_tokens_generated / total_time
    print("-" * 60)
    print(f"📊 性能统计:")
    print(f"- 总生成 Token 数量: {total_tokens_generated} tokens")
    print(f"- 端到端生成耗时  : {total_time:.2f} 秒")
    print(f"- 系统吞吐量 (Throughput): {throughput:.2f} tokens/s")
    print("-" * 60)

if __name__ == "__main__":
    main()