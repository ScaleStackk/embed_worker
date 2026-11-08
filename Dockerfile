# syntax=docker/dockerfile:1.7
FROM python:3.12-slim-bookworm

ARG EMBED_MODEL=BAAI/bge-small-en-v1.5
ENV EMBED_MODEL=${EMBED_MODEL} \
    MODEL_CACHE=/models \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --requirement requirements.txt

COPY download_model.py ./
RUN HF_HUB_OFFLINE=0 python download_model.py

COPY embed_judgment.py embedding_server.py import_qdrant.py container_entrypoint.py ./

RUN useradd --create-home --uid 10001 worker \
    && chown -R worker:worker /models /app

USER worker
WORKDIR /data
VOLUME ["/data"]
ENTRYPOINT ["python", "/app/container_entrypoint.py"]
