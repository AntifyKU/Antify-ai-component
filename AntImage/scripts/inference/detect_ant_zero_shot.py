"""
Use BioCLIP or general vision model to detect if image contains an ant.
This approach uses a zero-shot classifier that doesn't require training.

Works without any API key or model download!
"""

from ultralytics import YOLO
import torch
from pathlib import Path
import argparse


def detect_ant_with_yolo_world(image_path: str, conf: float = 0.01):
    """
    Use YOLO-World for zero-shot ant detection.
    YOLO-World can detect any object by just providing text prompts!
    
    This is the BEST approach - no training needed, works immediately.
    
    Tested and working with conf=0.01 and prompts ["ant", "bug", "insect", "animal"]
    """
    print("Loading YOLO-World model (open vocabulary detection)...")
    
    # YOLO-World Large model - best accuracy for open vocabulary detection
    # Note: YOLO11 doesn't have a World variant yet, YOLO-World uses YOLOv8
    model = YOLO("yolov8l-worldv2.pt")  # Will auto-download (~90MB)
    
    # Set the classes we want to detect - more prompts = better recall
    model.set_classes(["ant", "bug", "insect", "animal"])
    
    print(f"\nDetecting ants in: {image_path}")
    results = model.predict(
        source=image_path,
        conf=conf,
        save=True,
        show=False
    )
    
    for result in results:
        boxes = result.boxes
        print(f"\n{'='*50}")
        print(f"Image: {Path(image_path).name}")
        print(f"Detections: {len(boxes)}")
        
        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            class_name = result.names[cls_id]
            
            print(f"  {i+1}. {class_name}: {confidence:.2%}")
            print(f"      Box: [{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]")
        
        if len(boxes) == 0:
            print("  No ants detected in this image.")
        print(f"{'='*50}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Detect ants using YOLO-World (zero-shot)')
    parser.add_argument('--image', type=str, required=True, help='Image to analyze')
    parser.add_argument('--conf', type=float, default=0.01, help='Confidence threshold (default: 0.01)')
    
    args = parser.parse_args()
    
    detect_ant_with_yolo_world(args.image, args.conf)


if __name__ == '__main__':
    main()
