"""Main script for exposing the FastAPI API for YOLO inference"""
import os
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from typing import Dict, Any, Union

from fastapi import FastAPI, Query, File, UploadFile
from fastapi.responses import HTMLResponse, Response
from app.model.model import inference_on_img, inference_on_path
from app.model.model import __version__ as model_version

app = FastAPI()

@app.get('/')
def home() -> Dict[str, Union[int, Dict[str, Any]]]:
    """Ping method for checking API status"""
    return {
        'status_code': 200,
        'data': {
            'health_check': 'OK', 
            'model_version': model_version
        }}

@app.get('/images')
def get_available_images(
    path: str = Query(..., description='The path to search for images')
) -> Dict[str, Union[int, Dict[str, Any]]]:
    """Returns the list of available PNG and JPG image files in the path received as query string"""
    available_images = []
    path = path.replace('$', '/')   # Parse the path, as it has $ instead of /

    # Retrieve all images available for inference
    if os.path.exists(path):
        available_images = [filename for filename in os.listdir(path) if filename.endswith(('.png', '.jpg'))]
        return {
            'status_code': 200,
            'data': {
                'available_images': available_images, 
                'size': len(available_images)
        }}

    return {
        'status_code': 400, 
        'data': {
            'message': 'The provided path does not exist', 
            'size': 0
    }}

@app.get('/detect')
def detect(
    path: str = Query(..., description='The path to search for images')
) -> Dict[str, Union[int, Dict[str, Any]]]:
    """Performs YOLO inference on all the images available in the path received as query string"""
    path = path.replace('$', '/')   # Parse the path, as it has $ instead of /

    # Perform inference for the images on the given path 
    try:
        inference_results_data = inference_on_path(imgs_path=path)
    except Exception as err:
        print(f'An error occurred while trying to perform inference. {err}')
        return {
            'status_code': 500,
            'data': {}
        }

    return {
        'status_code': 200, 
        'data': {
            'inference_results': inference_results_data
    }}

@app.post('/detect_img')
async def detect_img(
    img: UploadFile = File(...)
) -> Dict[str, Union[int, Dict[str, Any]]]:
    """Performs YOLO inference on the received image"""
    if not img:
        return {
            'status_code': 400, 
            'data': {
                'message': 'No upload file sent'
        }}

    # Load the received image as PIL Image
    img_content = await img.read()
    image_stream = BytesIO(img_content)
    image = Image.open(image_stream)

    # Perform inference for the received image
    try:
        inference_results_data = inference_on_img(img=image)
    except Exception as err:
        print(f'An error occurred while trying to perform inference. {err}')
        return {
            'status_code': 500,
            'data': {}
        }

    return {
        'status_code': 200, 
        'data': {
            'image_name': img.filename,
            'image_size': img.size,
            'inference_results': inference_results_data
    }}

""" simple HTML forms to upload an image via GUI """
@app.get("/upload_form", response_class=HTMLResponse, include_in_schema=False)
async def root():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
</head>
<body>
<form action="/detect_img" name="form1" id="upload-image-form" method="POST" enctype="multipart/form-data">
    <!-- Selector to choose either the /detect_img or /detect_img_visual endpoint -->
    <select id="form_action" name="form_action" onchange="document.form1.action = this.value;">
    <option value="/detect_img" selected="selected">Detect (JSON output)</option>
    <option value="/detect_img_visual">Detect (PNG output)</option>
    </select>
    
    <!-- Simple file input -->
    <div class="form-group">
        <label for="image">Upload Image:</label>
        <input type="file" id="image" name="img" accept="image/png, image/jpeg, image/gif">
    </div>

    <button type="submit" >Submit</button>
</form>
</body>
</html>'''

""" drawing the result of the inference on the image then return it (for visual inspection) """
@app.post('/detect_img_visual', responses = {
        200: {
            "content": {"image/png": {}}
        }
    },response_class=Response)
async def mirror_img(
    img: UploadFile = File(...)
) -> Dict[str, Union[int, Dict[str, Any]]]:
    """Performs YOLO inference on the received image"""
    if not img:
        return {
            'status_code': 400, 
            'data': {
                'message': 'No upload file sent'
        }}

    # Load the received image as PIL Image
    img_content = await img.read()
    image_stream = BytesIO(img_content)
    image = Image.open(image_stream)

    # Perform inference for the received image
    try:
        inference_results_data = inference_on_img(img=image)
    except Exception as err:
        print(f'An error occurred while trying to perform inference. {err}')
        return {
            'status_code': 500,
            'data': {}
        }

    # ImageDraw allow to draw rectangle and text on the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(20)
    # For each result, draw the bounding box and write the class name and confidence
    for result in inference_results_data:
        draw.rectangle(list(result['box'].values()),outline="green")
        draw.text(xy=[result['box']['x1'],result['box']['y1'] +30], text="{} ({:.1f})".format(result['name'], 100*result['confidence']), font=font, fill="green" )

    # Generate the output as a bytes stream of an PNG image.
    output_stream = BytesIO()
    image.save(output_stream, 'PNG')

    return Response (content=output_stream.getvalue(), media_type="image/png")
