# Face Mask Detection MLOps Project

## Overview

This project demonstrates an end-to-end MLOps workflow for face mask detection using YOLO models exported to ONNX format.

The project includes:

* Model inference using ONNX Runtime
* Support for multiple YOLO models
* Command-line prediction module
* FastAPI-based inference service
* Dockerized deployment
* Docker Hub image registry
* Kubernetes orchestration using K3s
* Traefik ingress routing
* MLflow experiment tracking
* Streamlit-based web demo application

Future work may include CI/CD automation, Kubeflow integration.

---

## Project Structure

```text
.
├── api/
├── data/
│   └── samples/
├── deployment/
├── models/
│   ├── yolo11/
│   └── yolo26/
├── notebooks/
├── reports/
├── src/
│   ├── demo/
│   ├── inference/
│   └── tracking/
├── Dockerfile
├── Dockerfile.streamlit
├── requirements.txt
└── README.md
```

---

## Available Models

| Model   | Format |
| ------- | ------ |
| YOLO26n | ONNX   |
| YOLO11s | ONNX   |

Models are stored in the `models/` directory.

---

## Local Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Command Line Inference

Run prediction using the default model:

```bash
python src/inference/predict.py
```

Run prediction with a custom image:

```bash
python src/inference/predict.py --image test.png
```

Run prediction with a specific model:

```bash
python src/inference/predict.py --image test.png --model yolo11s
```

Prediction outputs are stored in:

```text
reports/predictions/
```

---

## FastAPI Inference Service

Start the API server:

```bash
uvicorn api.main:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

### Available Endpoints

| Method | Endpoint | Description |
|----------|----------|----------|
| GET | / | Health check |
| GET | /models | List available models |
| POST | /predict | Run inference |


### Predict Endpoint

```http
POST /predict
```

Parameters:

| Parameter | Description                   |
| --------- | ----------------------------- |
| file      | Input image                   |
| model     | Model name (default: yolo26n) |

Example:

```http
POST /predict?model=yolo11s
```

Response:

```json
{
  "timestamp": "...",
  "image_name": "...",
  "model_name": "...",
  "num_detections": 3,
  "detections": [...]
}
```

---

## Docker

Build image:

```bash
docker build -t facemask-api:v1 .
```

Run container:

```bash
docker run -p 8000:8000 facemask-api:v1
```

Access API:

```text
http://localhost:8000/docs
```

---

# Docker Hub

Image is published on Docker Hub:

```text
ngdangkhoa/facemask-api:v1
```

Pull image:

```bash
docker pull ngdangkhoa/facemask-api:v1
```

---

## Kubernetes (K3s)

Namespace:

```bash
kubectl create namespace facemask
```

Deploy application:

```bash
kubectl apply -f deployment/deployment.yaml
```

Create service:

```bash
kubectl apply -f deployment/service.yaml
```

Create ingress:

```bash
kubectl apply -f deployment/ingress.yaml
```

Check resources:

```bash
kubectl get all -n facemask
```
---

## Streamlit Demo Application

The project includes a lightweight Streamlit frontend for demonstrating model inference.

### Features

* Upload image files (`jpg`, `jpeg`, `png`)
* Select available models dynamically from the API
* Run inference using deployed FastAPI service
* Visualize detected bounding boxes
* Display prediction metadata and detection results

### Run Locally

```bash
streamlit run src/demo/app.py
```

Open:

```text
http://localhost:8501
```

### Docker

Build image:

```bash
docker build -f Dockerfile.streamlit -t facemask-demo:latest .
```

Run container:

```bash
docker run -p 8501:8501 \
  -e API_URL=http://localhost:8000 \
  facemask-demo:latest
```

### Docker Hub

```text
ngdangkhoa/facemask-demo:v2
```

Pull image:

```bash
docker pull ngdangkhoa/facemask-demo:v2
```

---

## Streamlit Kubernetes Deployment

### Streamlit Deployment

```bash
kubectl apply -f deployment/deployment-streamlit.yaml
kubectl apply -f deployment/service-streamlit.yaml
```

### Ingress

```bash
kubectl apply -f deployment/ingress.yaml
```

### Verify Resources

```bash
kubectl get all -n facemask
```

---

## Deployment Architecture

```text
                     Browser
                         │
                         ▼
                 Traefik Ingress
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼                                 ▼
   Streamlit UI                    FastAPI API
        │                                 │
        └──────────────┬──────────────────┘
                       ▼
                 ONNX Runtime
                       │
                       ▼
                YOLO ONNX Models
```

---


## MLflow Experiment Tracking

MLflow is used to track training experiments and store model-related artifacts.

### Start MLflow Server

On the Linux server:

```bash
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts \
  > mlflow.log 2>&1 &
```

Access MLflow UI:

```text
http://<SERVER_IP>:5000
```

To stop MLflow server:
```text
pkill -f "mlflow server"
```

### Log Existing Training Results

The project includes a tracking utility that automatically logs the latest training run of a model family.

Example:

```bash
python src/tracking/tracking.py --family yolo26
```

The script automatically:

1. Finds the latest training directory (e.g. `reports/yolo26/train4`)
2. Reads training parameters from `args.yaml`
3. Logs all training metrics from `results.csv`
4. Uploads generated training artifacts
5. Uploads the exported ONNX model

### Logged Information

Parameters:

* Training configuration from `args.yaml`
* Hyperparameters
* Model settings

Metrics:

* Training loss history
* Validation loss history
* Precision
* Recall
* mAP50
* mAP50-95

Artifacts:

* `results.csv`
* `results.png`
* `confusion_matrix.png`
* `confusion_matrix_normalized.png`
* Precision-Recall curves
* F1 curves
* Exported ONNX model

---
## Demo Endpoints

### FastAPI Swagger

```text
http://192.168.28.30/docs
```

### Streamlit Demo
```text
http://192.168.28.30/
```

### MLflow Tracking UI
```text
http://192.168.28.30/5000
```

---

## MLOps Workflow

```text
YOLO Training
      │
      ▼
Training Reports
      │
      ▼
Export ONNX Model
      │
      ▼
Docker Image
      │
      ▼
Docker Hub
      │
      ▼
Kubernetes (K3s)
      │
      ├────────► FastAPI Service
      │
      └────────► Streamlit Demo
```

---

## Technologies

* Python
* YOLO
* ONNX Runtime
* OpenCV
* FastAPI
* Streamlit
* Docker
* Docker Hub
* Kubernetes (K3s)
* Traefik Ingress
* MLflow

---

## Roadmap

### Completed

* [x] Inference module
* [x] FastAPI application
* [x] Image upload endpoint
* [x] Multi-model inference
* [x] Docker containerization
* [x] Docker Hub integration
* [x] Linux server deployment
* [x] K3s installation
* [x] Kubernetes Deployment
* [x] Kubernetes Service
* [x] Traefik Ingress
* [x] MLflow experiment tracking
* [x] Streamlit demo application
* [x] Streamlit deployment on Kubernetes

### Future improvements

* [ ] MLflow Model Registry
* [ ] CI/CD pipeline
* [ ] Automated retraining workflow

---