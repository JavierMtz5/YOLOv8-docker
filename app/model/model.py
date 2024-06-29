import os
from typing import Dict, List, Any
import json
from PIL import Image

import ultralytics
import ultralytics.engine
import ultralytics.engine.results

__version__ = "0.0.3"

model = ultralytics.YOLO(os.environ.get('YOLOV8_MODEL_DIR'))

def inference_on_path(imgs_path: str) -> List[Dict[str, Any]]:
    """
    Performs inference on the trained model for the YOLOv8 architecture 
    on the images available in the given path
    """
    results: ultralytics.engine.results.Results = model(source=imgs_path, show=False, conf=0.45)
    results_data = []
    for result in results:
        result_metadata = {
            'shape': result.orig_shape,
            'path': result.path,
            'detections': json.loads(result.tojson())
        }
        results_data.append(result_metadata)

    return results_data

def inference_on_img(img: Image) -> List[Dict[str, Any]]:
    """
    Performs inference on the trained model for the YOLOv8 architecture 
    on the given image
    """
    results: ultralytics.engine.results.Results = model(source=img, show=False, conf=0.45)
    result_data = []
    for result in results:
        result_data = json.loads(result.tojson())

    return result_data
