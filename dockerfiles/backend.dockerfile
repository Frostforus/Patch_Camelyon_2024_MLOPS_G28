# Base image
FROM python:3.11-slim
LABEL authors="konrad"

EXPOSE $PORT

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY app/backend/ app/backend/

COPY patch_camelyon_2024_mlops_g28/models app/backend/models

WORKDIR app/backend/
RUN pip install -r requirements.txt --no-cache-dir

CMD exec uvicorn main:app --port 8080 --host 0.0.0.0 --workers 1
