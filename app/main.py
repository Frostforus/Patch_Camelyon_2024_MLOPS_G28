import yaml
from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from pydantic import BaseModel
import os
from google.cloud import storage

# Specify the path to your YAML file
yaml_file_path = 'config.yaml'
# Load configurations from the YAML file
with open(yaml_file_path, 'r') as file:
    configs = yaml.safe_load(file)


class PredictionModel:
    def __init__(self, source_bucket_name=configs['google_cloud']['bucket_name'],
                 source_model_blob_name=configs['google_cloud']['model_blob_name'],
                 destination_file_name=configs['local_model_file_name'],
                 key_file_path=configs['google_cloud']['key_file_name']):
        self.model = None
        self.source_bucket_name = source_bucket_name
        self.source_model_blob_name = source_model_blob_name
        self.destination_file_name = destination_file_name
        self.key_file_path = key_file_path

    # TODO: add parameters to init function too
    def _download_model_from_blob(self):
        """Downloads a blob from the bucket."""
        storage_client = storage.Client.from_service_account_json(self.key_file_path)
        bucket = storage_client.get_bucket(self.source_bucket_name)
        blob = bucket.blob(self.source_model_blob_name)

        print(f"Started download of {self.source_model_blob_name} to {self.destination_file_name}.")
        blob.download_to_filename(self.destination_file_name)
        print(f"Model {self.source_model_blob_name} downloaded to {self.destination_file_name}.")

    def _load_model(self, force_download=False):
        # List model files in the current directory
        files = [model_file_name for model_file_name in os.listdir(".") if model_file_name.endswith(".ckpt")]
        print(files)
        if len(files) == 0 or force_download:
            print("No local model found. Downloading model from GCS.")
            self._download_model_from_blob()

            files.append(self.destination_file_name)

        else:
            print(f"Local model found, loading from file: {files[0]}")

        return files[0]

    def predict(self, image = None):
        return "Prediction with " + self.model


# Global variables,
app = FastAPI()
prediction_model = PredictionModel()

@app.post("/predict/test/")
async def prediction_test():
    msg = prediction_model.predict()
    return {"message": "Testing with dummy image. " + msg}


@app.post("/predict/")
async def prediction():
    print("File uploaded, proceeding with prediction")
    return {"message": "Image uploaded, resized, and saved successfully"}
