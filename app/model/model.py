from typing import Dict, List, Any
import json
from PIL import Image

import ultralytics
import ultralytics.engine
import ultralytics.engine.results

__version__ = "1.0.0"

model = ultralytics.YOLO('yolov8m.pt')

def inference_on_path(imgs_path: str) -> List[Dict[str, Any]]:
    """
    Runs inference on the YOLOv8 architecture for the images available in the given path
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
    Runs inference on the YOLOv8 architecture for the given image
    """
    results: ultralytics.engine.results.Results = model(source=img, show=False, conf=0.45)
    result_data = json.loads(results[0].tojson())

    return result_data
