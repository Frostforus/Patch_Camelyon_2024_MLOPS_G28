# Base image
FROM python:3.11-slim

# Set WandB API Key
ENV WANDB_API_KEY=792ab2b5bc699fe2e350a54f40aff67f76f00304

# Install DVC
RUN pip install dvc

# Initialize DVC
RUN dvc init

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy DVC-related files
COPY dvc.lock dvc.lock
COPY dvc.yaml dvc.yaml

RUN dvc pull -r gs://ml-ops-data-bucket/ -d /data


COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY patch_camelyon_2024_mlops_g28/ patch_camelyon_2024_mlops_g28/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

EXPOSE 8080

ENTRYPOINT ["python", "-u", "patch_camelyon_2024_mlops_g28/train_model.py"]
