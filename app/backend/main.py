import yaml
from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from pydantic import BaseModel
import os
from google.cloud import storage
import cv2
import numpy as np


class PredictionModel:
    def __init__(self,
                 force_download=False,
                 yaml_file_path='./config/config.yaml'
                 ):
        self.model = None

        # Load configurations from the YAML file
        with open(yaml_file_path, 'r') as file:
            configs = yaml.safe_load(file)
            self.source_bucket_name = configs['google_cloud']['bucket_name']
            self.source_model_blob_name = configs['google_cloud']['model_blob_name']
            self.destination_file_name = configs['local_model_file_name']
            self.key_file_path = configs['google_cloud']['key_file_name']
            self.target_size = tuple((configs['target_size']['height'], configs['target_size']['width']))


        self.model = self._load_model(force_download)

    # TODO: add parameters to init function too
    def _download_model_from_blob(self):
        """Downloads a blob from the bucket."""
        storage_client = storage.Client.from_service_account_json(self.key_file_path)
        bucket = storage_client.get_bucket(self.source_bucket_name)
        blob = bucket.blob(self.source_model_blob_name)

        print(f"Started download of {self.source_model_blob_name} to {self.destination_file_name}.")
        blob.download_to_filename(self.destination_file_name)
        print(f"Model {self.source_model_blob_name} downloaded to {self.destination_file_name}.")

    def _load_model(self, force_download):
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

    def _resize_image(self, image: UploadFile):

        contents = image.file.read()
        image_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        return cv2.resize(image_array, self.target_size)


    def predict(self, image: UploadFile=None):
        if image:
            self._resize_image(image)
        return "Prediction with " + self.model


# Global variables,
app = FastAPI()
prediction_model = PredictionModel()


@app.post("/predict/test/")
async def prediction_test():
    msg = prediction_model.predict()
    return {"message": "Testing with dummy image. " + msg}


@app.post("/predict/")
async def predict_from_image(image: UploadFile = File(...)):
    print("File uploaded, proceeding with prediction")
    msg = prediction_model.predict(image)

    return {"message": "Testing with uploaded image. " + msg}
