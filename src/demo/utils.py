from pathlib import Path
from io import BytesIO
import os
import cv2
import numpy as np
import requests
from PIL import Image

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

    
def get_available_models():
    """
    Automatically discover available ONNX models
    from the models directory.
    """
    
    response = requests.get(
        f"{API_URL}/models",
        timeout=30
    )

    response.raise_for_status()

    return response.json()["models"]


def call_prediction_api(uploaded_file, model_name):
    """
    Send image to FastAPI prediction endpoint.
    """

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type
        )
    }

    response = requests.post(
        f"{API_URL}/predict",
        params={"model": model_name},
        files=files,
        timeout=120
    )

    response.raise_for_status()

    return response.json()



def draw_detections(image, detections):

    image_np = np.array(image)

    image_bgr = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )


    for det in detections:

        bbox = det["bbox"]

        x1 = int(bbox["x1"])
        y1 = int(bbox["y1"])
        x2 = int(bbox["x2"])
        y2 = int(bbox["y2"])

        cls_name = det["class_name"]
        if cls_name == "with_mask":
            color = (255, 0, 0)       # blue
        else:
            color = (255, 200, 0)     # light blue
        conf = det["confidence"]

        label = f"{cls_name} {conf:.2f}"

        # bbox mảnh giống YOLO
        cv2.rectangle(
            image_bgr,
            (x1, y1),
            (x2, y2),
            color,
            1
        )

        # kích thước chữ nhỏ
        font_scale = 0.35
        thickness = 1

        (tw, th), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            thickness
        )

        # nền label
        cv2.rectangle(
            image_bgr,
            (x1, y1 - th - 4),
            (x1 + tw + 2, y1),
            color,
            -1
        )

        # chữ màu trắng
        cv2.putText(
            image_bgr,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(image_rgb)