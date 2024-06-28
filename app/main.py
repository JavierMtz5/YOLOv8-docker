import os
from typing import List

from fastapi import FastAPI, Query
from pydantic import BaseModel
from app.model.model import inference
from app.model.model import __version__ as model_version

app = FastAPI()

@app.get("/")
def home():
    return {'status_code': 200, "health_check": "OK", "model_version": model_version}

@app.get("/images")
def get_available_images(path: str = Query(..., description="The path to search for images")):
    available_images = []
    # Parse the path, as it has $ instead of /
    path = path.replace('$', '/')
    # Retrieve all images available for inference
    if os.path.exists(path):
        available_images = [filename for filename in os.listdir(path) if filename.endswith(('.png', '.jpg'))]
        return {'status_code': 200, 'available_images': available_images, 'size': len(available_images)}
    return {'status_code': 400, 'message': 'The provided path does not exist', 'size': 0}

@app.get("/detect")
def detect(path: str = Query(..., description="The path to search for images")):
    # Parse the path, as it has $ instead of /
    path = path.replace('$', '/')
    try:
        inference_results_data = inference(path)
    except Exception as err:
        print(f'An error occurred while trying to perform inference. {err}')
        return {'status_code': 500}

    return {'status_code': 200, "inference_results": inference_results_data}