#!/bin/bash
# ================================================================
# UniAssist - Lightweight Models Setup for Low-RAM Server
# ================================================================

OLLAMA_HOST="${OLLAMA_HOST:-localhost:11434}"

echo "========================================================"
echo "UniAssist - Pulling Lightweight Models into Ollama"
echo "Target Ollama Host: $OLLAMA_HOST"
echo "========================================================"

# Wait for Ollama to be responsive
echo "Checking connection to Ollama..."
until curl -s "http://$OLLAMA_HOST/api/tags" > /dev/null; do
    echo "Waiting for Ollama service to be ready on $OLLAMA_HOST..."
    sleep 3
done
echo "Ollama service is up!"

# 1. Model 1: Qwen 2.5 0.5B (Default - 390MB)
echo ""
echo "--> [1/3] Pulling Qwen 2.5 0.5B (Ideal for 1GB RAM EC2 instances)..."
curl -X POST "http://$OLLAMA_HOST/api/pull" -d '{"name": "qwen2.5:0.5b"}'

# 2. Model 2: TinyLlama 1.1B (Fast - 630MB)
echo ""
echo "--> [2/3] Pulling TinyLlama 1.1B (Fast generation)..."
curl -X POST "http://$OLLAMA_HOST/api/pull" -d '{"name": "tinyllama"}'

# 3. Model 3: Qwen 2.5 1.5B (Higher reasoning - 980MB)
echo ""
echo "--> [3/3] Pulling Qwen 2.5 1.5B (Balanced reasoning under 1.5GB RAM)..."
curl -X POST "http://$OLLAMA_HOST/api/pull" -d '{"name": "qwen2.5:1.5b"}'

echo ""
echo "========================================================"
echo "All 3 lightweight models pulled successfully!"
echo "Available models in Ollama:"
curl -s "http://$OLLAMA_HOST/api/tags" | grep -o '"name":"[^"]*"'
echo "========================================================"
