# client.py
import requests
import base64

filename = "/home/jmartinez/ml_app/data/small_coco_dataset/images/train2017/000000000328.jpg"
files = {"img": (filename, open(filename, 'rb'), "image/jpeg")}
response = requests.post(
    'http://127.0.0.1:8080/detect_img',
    files=files
)
print(response.json())
