# Base image
FROM python:3.11-slim
LABEL authors="konrad"

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY app/backend/ app/backend/

WORKDIR app/backend/
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-m", "uvicorn app.backend.main:app", "--reload","--port 8000"]
