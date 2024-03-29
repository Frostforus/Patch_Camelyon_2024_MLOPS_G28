# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY patch_camelyon_2024_mlops_g28/ patch_camelyon_2024_mlops_g28/
COPY models/ models/
COPY data/processed/test_prediction_images.pt data/processed/test_prediction_images.pt

WORKDIR /
RUN mkdir data/predictions/
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "patch_camelyon_2024_mlops_g28/predict_model.py", "models/test_model.pt", "data/processed/test_prediction_images.pt"]
