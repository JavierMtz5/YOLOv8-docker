import os
from typing import Dict, List, Any
import json
from PIL import Image
from pathlib import Path

import ultralytics
import ultralytics.engine
import ultralytics.engine.results

__version__ = "0.0.3"

model = ultralytics.YOLO(os.environ.get('YOLOV8_MODEL_DIR'))

def inference_on_path(imgs_path: str) -> List[Dict[str, Any]]:
    """Performs inference on the trained model for the YOLOv8 architecture"""
    results: ultralytics.engine.results.Results = model(source=imgs_path, show=False, conf=0.45)
    json_results = []
    for result in results:
        result_metadata = {
            'shape': result.orig_shape,
            'path': result.path,
            'detections': json.loads(result.tojson())
        }
        json_results.append(result_metadata)

    return json_results

def inference_on_img(img: Image) -> List[Dict[str, Any]]:
    """Performs inference on the trained model for the YOLOv8 architecture"""
    results: ultralytics.engine.results.Results = model(source=img, show=False, conf=0.45)
    result_data = []
    for result in results:
        result_data = json.loads(result.tojson())

    return result_data
