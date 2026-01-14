"""
Use pre-trained ant detection model from Roboflow Universe.
This script downloads and uses a YOLOv8 model trained to detect ants with bounding boxes.

Usage:
    # First time: Install roboflow
    pip install roboflow

    # Download model and run detection
    python detect_ant_pretrained.py --image path/to/image.jpg
    
    # Or use Roboflow Inference (faster, no download needed)
    python detect_ant_pretrained.py --image path/to/image.jpg --use-api
"""

import argparse
from pathlib import Path


def download_roboflow_model(api_key: str = None):
    """
    Download pre-trained ant detection model from Roboflow.
    
    Note: You may need a free Roboflow API key from https://roboflow.com
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Please install roboflow: pip install roboflow")
        return None
    
    # Use public model - may work without API key for some models
    rf = Roboflow(api_key=api_key or "YOUR_API_KEY")
    
    # Popular ant detection projects on Roboflow Universe:
    # 1. steam-re/ant-detection - 500 images
    # 2. search "ant detection" on universe.roboflow.com for more
    
    project = rf.workspace().project("ant-detection")
    model = project.version(1).model
    
    return model


def detect_with_ultralytics(model_path: str, image_path: str, conf: float = 0.25):
    """
    Run detection using downloaded YOLO model.
    """
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    results = model.predict(
        source=image_path,
        conf=conf,
        save=True,
        show=False
    )
    
    for result in results:
        boxes = result.boxes
        print(f"\nDetections in {Path(image_path).name}:")
        print(f"Found {len(boxes)} ant(s)")
        
        for i, box in enumerate(boxes):
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"  {i+1}. Confidence: {conf:.2%}, Box: [{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]")
    
    return results


def detect_with_roboflow_api(image_path: str, api_key: str, model_id: str = "ant-detection/1"):
    """
    Run detection using Roboflow Inference API (no model download needed).
    Requires API key from roboflow.com
    """
    try:
        from inference import get_model
    except ImportError:
        print("Please install inference: pip install inference")
        return None
    
    model = get_model(model_id=model_id, api_key=api_key)
    results = model.infer(image_path)
    
    print(f"\nDetections in {Path(image_path).name}:")
    if hasattr(results, 'predictions'):
        predictions = results.predictions
        print(f"Found {len(predictions)} ant(s)")
        for i, pred in enumerate(predictions):
            print(f"  {i+1}. {pred.class_name}: {pred.confidence:.2%}")
    
    return results


def use_insect_detector():
    """
    Alternative: Use a general insect detection model.
    These models can detect various insects including ants.
    """
    print("\n=== Alternative Insect Detection Models ===")
    print("""
Available pre-trained models on Roboflow Universe:
    
1. Ant Detection (Steam RE)
   - URL: https://universe.roboflow.com/steam-re/ant-detection
   - 500+ images, YOLOv5/v8 models
   
2. Insect Detection 
   - URL: https://universe.roboflow.com/search?q=insect+detection
   - Various models for wasps, bees, flies, ants
   
3. Fire Ant Detection (from aerial images)
   - Specialized for fire ant mounds

To use these models:
1. Go to the Roboflow Universe URL
2. Click "Model" tab
3. Try the model in browser, or
4. Get your API key and download the model

Example code to use Roboflow model:
    
    from roboflow import Roboflow
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace().project("ant-detection")  
    model = project.version(1).model
    prediction = model.predict("your_image.jpg", confidence=40)
    prediction.save("prediction.jpg")
""")


def main():
    parser = argparse.ArgumentParser(description='Detect ants using pre-trained model')
    parser.add_argument('--image', type=str, help='Image path to detect ants in')
    parser.add_argument('--model', type=str, help='Path to local YOLO model (.pt file)')
    parser.add_argument('--api-key', type=str, help='Roboflow API key')
    parser.add_argument('--use-api', action='store_true', help='Use Roboflow API instead of local model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--info', action='store_true', help='Show info about available models')
    
    args = parser.parse_args()
    
    if args.info:
        use_insect_detector()
        return
    
    if not args.image:
        print("Please provide an image path with --image")
        print("Or run with --info to see available models")
        use_insect_detector()
        return
    
    if args.use_api:
        if not args.api_key:
            print("API key required for Roboflow API. Get one at https://roboflow.com")
            return
        detect_with_roboflow_api(args.image, args.api_key)
    elif args.model:
        detect_with_ultralytics(args.model, args.image, args.conf)
    else:
        print("Please provide either --model (local .pt file) or --use-api with --api-key")
        use_insect_detector()


if __name__ == '__main__':
    main()
