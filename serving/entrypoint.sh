#!/bin/bash
# Start vLLM in the background, wait for it, then launch FastAPI.
set -e

MODEL_DIR=${MODEL_DIR:-/model}
VLLM_PORT=${VLLM_PORT:-8001}
API_PORT=${API_PORT:-8000}

echo "Starting vLLM on port $VLLM_PORT …"
vllm serve "$MODEL_DIR" \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --port "$VLLM_PORT" \
    --host 0.0.0.0 &

VLLM_PID=$!

# Wait until vLLM is ready
echo "Waiting for vLLM to become ready …"
until curl -sf "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM is ready."

echo "Starting FastAPI on port $API_PORT …"
cd /app
exec uvicorn serving.app:app --host 0.0.0.0 --port "$API_PORT"
