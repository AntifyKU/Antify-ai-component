# Antify AI Component

🐜 **Ant Detection AI** - A YOLO-based model and REST API for detecting ants in images.

## Features

- **Ant Detection Model**: YOLO11s model trained on ant images
- **REST API**: FastAPI server for easy integration
- **High Accuracy**: 80%+ mAP50 on validation dataset

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Run the API

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 3. Test Detection

```bash
python test_api.py path/to/ant_image.jpg
```

Or open http://localhost:8000/docs for interactive API documentation.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/models` | List available models |
| POST | `/detect` | Detect ants → JSON response |
| POST | `/detect/visualize` | Detect ants → Annotated image |
| POST | `/classify` | Classify ant species (if model loaded) |

## Example Usage

### Python

```python
import requests

with open("ant_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect",
        files={"file": f}
    )

result = response.json()
print(f"Found {result['num_detections']} ants")
```

### cURL

```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@ant_image.jpg"
```

## Model Info

- **Architecture**: YOLO11s
- **Task**: Object Detection
- **Classes**: ant
- **Input Size**: 640x640

## Project Structure

```
├── api/
│   ├── app.py          # FastAPI server
│   └── __init__.py
├── models/
│   └── ant_detector_v1.pt  # Trained model
├── scripts/
│   ├── inference/      # Inference scripts
│   └── training/       # Training scripts
├── requirements_api.txt
├── test_api.py
└── README.md
```

## License

MIT License
