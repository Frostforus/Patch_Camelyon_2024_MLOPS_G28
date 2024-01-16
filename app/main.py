from fastapi import FastAPI, UploadFile, File
from http import HTTPStatus
from pydantic import BaseModel

app = FastAPI()

MODEL = None


def load_modeL_from_g_bucket():
    return "model"


MODEL = load_modeL_from_g_bucket()


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {"message": f"Hello {MODEL}"}
