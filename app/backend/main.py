import torch
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
                 yaml_file_path='./config/config.yaml'
                 ):
        self.model = None

        # Load configurations from the YAML file
        with open(yaml_file_path, 'r') as file:
            configs = yaml.safe_load(file)
            self.source_bucket_name = configs['google_cloud']['bucket_name']
            self.source_model_blob_name = configs['google_cloud']['model_blob_name']
            self.destination_dir_path = configs['local']['model_dir']
            self.destination_file_name = configs['local']['model_file_name']
            self.destination_file_path = self.destination_dir_path + '/' + self.destination_file_name
            self.key_file_path = configs['google_cloud']['key_file_name']
            self.model_input_dimensions = {"height": configs['model']['input_dimensions']['height'],
                                           "width": configs['model']['input_dimensions']['width'],
                                           "channels": configs['model']['input_dimensions']['channels']}
        try:
            self._load_model(configs['model']['force_download'])
        except Exception as e:
            print("Error occurred during model loading: ", e)

    def _download_model_from_blob(self):
        """Downloads a blob from the bucket."""
        storage_client = storage.Client.from_service_account_json(self.key_file_path)
        bucket = storage_client.get_bucket(self.source_bucket_name)
        blob = bucket.blob(self.source_model_blob_name)

        print(f"Started download of {self.source_model_blob_name} to {self.destination_file_path}.")
        blob.download_to_filename(self.destination_dir_path + '/' + self.destination_file_name)
        print(f"Model {self.source_model_blob_name} downloaded to {self.destination_file_path}.")

    def _load_model(self, force_download=False):
        # List model files in the current directory,
        files = [self.destination_dir_path + '/' + model_file_name for model_file_name in
                 os.listdir(self.destination_dir_path) if model_file_name.endswith(".pt")]
        if len(files) == 0 or force_download:
            print("No local model found. Downloading model from GCS.")
            self._download_model_from_blob()
        else:
            print(f"Local model found, loading from file: {self.destination_file_path}")

        # Load the model from local file, and enable eval mode, to disable randomness and dropout
        self.model = torch.load(self.destination_file_path)
        self.model.eval()

    def _resize_image(self, image: UploadFile):
        contents = image.file.read()
        image_array = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        return cv2.resize(image_array, self.model_input_dimensions)

    def predict(self, image: UploadFile = None):
        if image:
            image = self._resize_image(image)
            # Convert image to tensor and add batch dimension,
            # TODO: swtich this with the correct predict function
            prediction = self.model.predict(image)
            return prediction
        else:
            return "No image provided."


# Global variables,
app = FastAPI()
prediction_model = PredictionModel()


@app.post("/predict/test/")
async def prediction_test():
    msg = prediction_model.predict()
    return {"message": "Testing with dummy image. " + msg}


@app.post("/predict/")
async def predict_from_image(image: UploadFile = File(...)):
    try:
        print("File uploaded, proceeding with prediction")
        msg = prediction_model.predict(image)

        return {"message": "Prediction result: " + msg, "prediction": msg, "status": HTTPStatus.OK}
    except Exception as e:
        print("Error occurred during prediction: ", e)
        return {"message": "Error occurred during prediction: " + str(e), "status": HTTPStatus.IM_A_TEAPOT}


@app.get("/ping")
async def ping():
    return {"message": "pong", "status": HTTPStatus.OK}
