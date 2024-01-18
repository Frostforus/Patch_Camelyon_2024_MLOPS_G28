# Base image
FROM python:3.11-slim
LABEL authors="konrad"

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY app/backend/ app/backend/

WORKDIR app/backend/
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-m", "uvicorn app.backend.main:app", "--reload","--port 8000"]
