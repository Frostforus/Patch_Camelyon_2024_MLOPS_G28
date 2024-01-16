# Base image
FROM python:3.11-slim

COPY ml-ops-ex-b1b1bd7f7ce9.json ml-ops-ex-b1b1bd7f7ce9.json

ENV GOOGLE_APPLICATION_CREDENTIALS=ml-ops-ex-b1b1bd7f7ce9.json

# Set WandB API Key
ENV WANDB_API_KEY=792ab2b5bc699fe2e350a54f40aff67f76f00304

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY patch_camelyon_2024_mlops_g28/ patch_camelyon_2024_mlops_g28/
COPY data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "patch_camelyon_2024_mlops_g28/train_model.py"]
