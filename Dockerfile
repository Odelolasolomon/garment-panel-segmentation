FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_WEIGHTS=/app/weights/best.pt \
    SERVICE_NAME=panel-seg

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /usr/sbin/nologin appuser

COPY requirements.txt requirements-deploy.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements-deploy.txt

COPY model ./model
COPY predict.py apply_fabric.py benchmark_latency.py ./
COPY serve.py ./
COPY weights/.gitkeep ./weights/.gitkeep

RUN mkdir -p /app/weights /app/outputs \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).read()"

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]