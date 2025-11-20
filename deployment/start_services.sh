#!/bin/bash

# SFT-ner Service Startup Script
# This script starts the Base Model API, LoRA Model API, and Streamlit Demo

set -e

echo "🚀 Starting SFT-ner Services..."
echo "================================"

# Configuration
BASE_PORT=8003
LORA_PORT=8002
DEMO_PORT=8501
CUDA_DEVICE=1

cd /home/ubuntu/SFT-ner/military-ner-showcase

# Start Base Model API
echo "Starting Base Model API on port $BASE_PORT..."
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
nohup /home/ubuntu/venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/SFT-ner/military-ner-project/saves/Qwen3-4B \
  --served-model-name qwen3-base \
  --port $BASE_PORT \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 8192 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.42 \
  > /tmp/vllm_base.log 2>&1 &

echo "✓ Base Model started with PID: $!"

# Wait for Base model to load
sleep 30

# Start LoRA Model API
echo "Starting LoRA Model API on port $LORA_PORT..."
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
nohup /home/ubuntu/venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model /home/ubuntu/SFT-ner/military-ner-project/saves/Qwen3-4B \
  --served-model-name qwen3-base \
  --enable-lora \
  --lora-modules qwen3-ner-zero3=/home/ubuntu/SFT-ner/military-ner-project/saves/Qwen3-4B/lora/ner_zero3/ \
  --port $LORA_PORT \
  --host 0.0.0.0 \
  --trust-remote-code \
  --max-model-len 12288 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.46 \
  > /tmp/vllm_lora.log 2>&1 &

echo "✓ LoRA Model started with PID: $!"

# Wait for LoRA model to load
sleep 30

# Start Streamlit Demo
echo "Starting Streamlit Demo on port $DEMO_PORT..."
export BASE_API_URL=http://localhost:$BASE_PORT
export LORA_API_URL=http://localhost:$LORA_PORT
nohup /home/ubuntu/venv/bin/python3 -m streamlit run demo/app.py \
  --server.port $DEMO_PORT \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.fileWatcherType none \
  > /tmp/streamlit.log 2>&1 &

echo "✓ Streamlit Demo started with PID: $!"

echo ""
echo "================================"
echo "✅ All services started successfully!"
echo "Demo URL: http://IP_ADDRESS:$DEMO_PORT"
echo "Base Model API: http://IP_ADDRESS:$BASE_PORT"
echo "LoRA Model API: http://IP_ADDRESS:$LORA_PORT"
echo ""
echo "Logs:"
echo "  Base Model: tail -f /tmp/vllm_base.log"
echo "  LoRA Model: tail -f /tmp/vllm_lora.log"
echo "  Streamlit: tail -f /tmp/streamlit.log"
