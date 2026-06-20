from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path

from src.inference.predictor import predict_image

app = FastAPI(
    title="Face Mask Detection API",
    version="1.0.0"
)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# helpers

def resolve_model_path(model_name: str):

    matches = list(
        Path("models").rglob(
            f"{model_name}.onnx"
        )
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Không tìm thấy model: {model_name}"
        )

    if len(matches) > 1:
        raise RuntimeError(
            f"Có nhiều model trùng tên: {model_name}"
        )

    return str(matches[0])

def get_available_models():

    model_files = Path("models").rglob("*.onnx")

    return sorted(
        model_file.stem
        for model_file in model_files
    )


# endpoints

@app.get("/")
def root():
    return {
        "message": "Face Mask Detection API"
    }

@app.get("/models")
def list_models():

    return {
        "models": get_available_models()
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = "yolo26n"):

    image_path = UPLOAD_DIR / file.filename
    # model_path = resolve_model_path(model)

    with open(image_path, "wb") as f:
        f.write(await file.read())

    try:
        model_path = resolve_model_path(model)

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not found"
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    result = predict_image(str(image_path), model_path)

    return result["result_json"]