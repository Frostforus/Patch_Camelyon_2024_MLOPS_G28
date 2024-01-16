from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from pydantic import BaseModel

app = FastAPI()

MODEL = None

from google.cloud import storage


def download_blob(bucket_name,
                  source_blob_name,
                  destination_file_name,
                  key_file_path='prediction_model_bucket_service_account_key_file.json'):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client.from_service_account_json(key_file_path)
    bucket = storage_client.get_bucket(bucket_name)

    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)


    print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")


# Replace these values with your own
bucket_name = 'prediction-model-bucket'
source_blob_name = 'helle_google.txt'
destination_file_name = 'local_file.txt'

download_blob(bucket_name, source_blob_name, destination_file_name)
