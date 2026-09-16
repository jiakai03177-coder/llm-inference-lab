#!/bin/bash
# ==============================================================================
# 第 9 周: vLLM OpenAI-Compatible API Server 启动脚本
# 模型: Qwen/Qwen2.5-0.5B-Instruct (本地缓存快照)
# 硬件: NVIDIA GeForce RTX 3060 Laptop (6GB VRAM)
# ==============================================================================

set -e

# 1. 关键环境配置 (适配 WSL2 与 CUDA 13)
export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WSL2_ENABLE_PIN_MEMORY=1

CU13_PATH="/home/asus/vllm-env/lib/python3.12/site-packages/nvidia/cu13/lib"
if [ -d "$CU13_PATH" ]; then
    export LD_LIBRARY_PATH="$CU13_PATH:$LD_LIBRARY_PATH"
fi

# 2. 模型路径与服务参数
SNAPSHOT_PATH="/home/asus/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
if [ -d "$SNAPSHOT_PATH" ]; then
    MODEL_PATH="$SNAPSHOT_PATH"
else
    MODEL_PATH="Qwen/Qwen2.5-0.5B-Instruct"
fi

HOST="0.0.0.0"
PORT=8000
SERVED_MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
GPU_MEM_UTIL=0.75
MAX_MODEL_LEN=2048

echo "================================================================="
echo "🚀 正在启动 vLLM OpenAI 兼容 API 服务器..."
echo "- 模型路径: $MODEL_PATH"
echo "- 服务名称: $SERVED_MODEL_NAME"
echo "- 监听地址: http://$HOST:$PORT"
echo "- 显存配额: $GPU_MEM_UTIL (RTX 3060 6GB 保护模式)"
echo "- 上下文长: $MAX_MODEL_LEN"
echo "================================================================="

# 3. 激活虚拟环境并启动服务
source /home/asus/vllm-env/bin/activate

exec vllm serve "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code

