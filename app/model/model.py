import os
from typing import Dict, Union, List, Any
from datetime import datetime
import json
from pathlib import Path

import ultralytics
import ultralytics.engine
import ultralytics.engine.results

__version__ = "0.0.3"

BASE_DIR = Path(__file__).resolve(strict=True).parent
INFERENCE_DIR = os.path.join(BASE_DIR, 'inference')
# TODO: Get from envar
model_weights_filename = 'coco_detector.pt'
MODEL_PATH = os.path.join(BASE_DIR, 'model_weights', model_weights_filename)

model = ultralytics.YOLO(MODEL_PATH)

def inference(imgs_path: str) -> List[Dict[str, Any]]:
    """Performs inference on the trained model for the YOLOv8 architecture"""
    results: ultralytics.engine.results.Results = model(source=imgs_path, show=False, save=True, conf=0.45)
    json_results = []
    for result in results:
        json_results.append(json.loads(result.tojson()))

    return json_results

# def parse_inference_results(results: ultralytics.engine.results.Results) -> List[Dict[str, Any]]:
#     """Converts the results obtained from the ultralytics inference"""
#     results_data = []
#     for result in results:
#         result_data = {'boxes': [],
#                        'image_shape': result.orig_shape,
#                        'image_path': result.path,
#                        'save_dir': result.save_dir}
#         boxes: ultralytics.engine.results.Boxes = result.boxes
#         for box in boxes:
#             box_data = {}
#             coordinates = box.xyxy.cpu().detach().numpy().flatten().tolist()
#             conf = box.conf
#             obj_class = box.cls
#             box_data['box_id'] = box.id
#             box_data['coordinates'] = {'x1': coordinates[0],
#                                         'x2': coordinates[1],
#                                         'y1': coordinates[2],
#                                         'y2': coordinates[3]}
#             box_data['class'] = obj_class
#             box_data['conf'] = conf
#             result_data['boxes'].append(box_data)

#         results_data.append(result_data)

#     return results_data
