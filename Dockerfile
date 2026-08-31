FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=5000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY outputs/cnn_gru_20260815_075909/final_model.pt ./outputs/cnn_gru_20260815_075909/final_model.pt
COPY data/processed/fusion/fusion_dataset.csv ./data/processed/fusion/fusion_dataset.csv

RUN mkdir -p /app/data/logs

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "src.api.app:app"]