# ── Text-to-SQL Serving Container ────────────────────────────────────────────
# Bundles vLLM + FastAPI into a single GPU-enabled container.
#
# Build:
#   docker build -t text2sql-serving serving/
#
# Run (requires --gpus all and the model checkpoint mounted):
#   docker run --gpus all \
#     -p 8000:8000 \
#     -v $(pwd)/training/outputs/best_checkpoint:/model \
#     -v $(pwd)/data/raw/spider/database:/spider_db \
#     -e SPIDER_DB_DIR=/spider_db \
#     text2sql-serving

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    vllm>=0.4.1 \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    httpx>=0.27.0 \
    pydantic>=2.7.0 \
    pyyaml>=6.0

# ── Application code ──────────────────────────────────────────────────────────
COPY configs/ configs/
COPY scripts/ scripts/
COPY serving/ serving/

# ── Startup script: launches vLLM then FastAPI ────────────────────────────────
COPY serving/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 8001

ENV MODEL_DIR=/model
ENV VLLM_URL=http://localhost:8001/v1
ENV SPIDER_DB_DIR=/spider_db

ENTRYPOINT ["/entrypoint.sh"]
