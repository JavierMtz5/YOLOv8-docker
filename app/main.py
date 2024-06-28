import os
import base64
from typing import List
from PIL import Image
from io import BytesIO

from fastapi import FastAPI, Query, File, UploadFile, Form
from pydantic import BaseModel
from app.model.model import inference_on_img, inference_on_path
from app.model.model import __version__ as model_version

app = FastAPI()

@app.get('/')
def home():
    return {
        'status_code': 200, 
        'health_check': 'OK', 
        'model_version': model_version
    }

@app.get('/images')
def get_available_images(path: str = Query(..., description='The path to search for images')):
    available_images = []
    # Parse the path, as it has $ instead of /
    path = path.replace('$', '/')
    # Retrieve all images available for inference
    if os.path.exists(path):
        available_images = [filename for filename in os.listdir(path) if filename.endswith(('.png', '.jpg'))]
        return {'status_code': 200, 'available_images': available_images, 'size': len(available_images)}
    return {
        'status_code': 400, 
        'message': 'The provided path does not exist', 
        'size': 0
    }

@app.get('/detect')
def detect(path: str = Query(..., description='The path to search for images')):
    # Parse the path, as it has $ instead of /
    path = path.replace('$', '/')
    try:
        inference_results_data = inference_on_path(imgs_path=path)
    except Exception as err:
        print(f'An error occurred while trying to perform inference. {err}')
        return {'status_code': 500}

    return {
        'status_code': 200, 
        'inference_results': inference_results_data
    }

@app.post('/detect_img')
async def detect_img(img: UploadFile = File(...)):
    if not img:
        return {'status_code': 400, 'message': 'No upload file sent'}

    img_content = await img.read()
    image_stream = BytesIO(img_content)
    image = Image.open(image_stream)
    inference_results_data = inference_on_img(img=image)
    return {
        'status_code': 200, 
        'image_name': img.filename,
        'image_size': img.size,
        'inference_results': inference_results_data
    }
