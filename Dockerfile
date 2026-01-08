FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY *.py ./

RUN pip install --upgrade pip \
    && pip install numpy requests pika python-dotenv

CMD ["python", "worker_score_apartments.py"]
