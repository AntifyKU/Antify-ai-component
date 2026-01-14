"""
Predict/detect ants in images using trained YOLOv8 model.

Usage:
    # Single image prediction
    python predict_ant.py --model runs/classify/ant_detector/weights/best.pt --image path/to/image.jpg
    
    # Batch prediction on directory
    python predict_ant.py --model runs/classify/ant_detector/weights/best.pt --source images/
    
    # Use webcam
    python predict_ant.py --model runs/classify/ant_detector/weights/best.pt --source 0
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2


def predict_classification(
    model_path: str,
    source: str,
    conf_threshold: float = 0.5,
    save: bool = True,
    show: bool = False
):
    """
    Run classification prediction on image(s).
    
    Args:
        model_path: Path to trained model weights
        source: Image path, directory, or video source
        conf_threshold: Confidence threshold for predictions
        save: Whether to save results
        show: Whether to display results
    """
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Running inference on: {source}")
    results = model.predict(
        source=source,
        conf=conf_threshold,
        save=save,
        show=show,
        verbose=True
    )
    
    # Print results for each image
    for result in results:
        probs = result.probs
        if probs is not None:
            top1_idx = probs.top1
            top1_conf = probs.top1conf.item()
            class_name = result.names[top1_idx]
            
            print(f"\n{'='*50}")
            print(f"Image: {Path(result.path).name}")
            print(f"Prediction: {class_name}")
            print(f"Confidence: {top1_conf:.2%}")
            print(f"{'='*50}")
            
            # Show top 5 predictions if available
            if hasattr(probs, 'top5'):
                print("\nTop predictions:")
                for idx, (class_idx, conf) in enumerate(zip(probs.top5, probs.top5conf)):
                    print(f"  {idx+1}. {result.names[class_idx]}: {conf:.2%}")
    
    return results


def predict_detection(
    model_path: str,
    source: str,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    save: bool = True,
    save_crop: bool = False,
    show: bool = False
):
    """
    Run object detection on image(s).
    
    Args:
        model_path: Path to trained model weights
        source: Image path, directory, or video source
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        save: Whether to save annotated results
        save_crop: Whether to save cropped detections
        show: Whether to display results
    """
    print(f"Loading detection model: {model_path}")
    model = YOLO(model_path)
    
    print(f"Running detection on: {source}")
    results = model.predict(
        source=source,
        conf=conf_threshold,
        iou=iou_threshold,
        save=save,
        save_crop=save_crop,
        show=show,
        verbose=True
    )
    
    # Print detection results
    for result in results:
        boxes = result.boxes
        print(f"\n{'='*50}")
        print(f"Image: {Path(result.path).name}")
        print(f"Detections: {len(boxes)}")
        
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            class_name = result.names[class_id]
            
            print(f"  {i+1}. {class_name} ({conf:.2%}) at [{int(xyxy[0])}, {int(xyxy[1])}, {int(xyxy[2])}, {int(xyxy[3])}]")
        
        print(f"{'='*50}")
    
    return results


def is_ant_present(model_path: str, image_path: str, threshold: float = 0.5) -> tuple:
    """
    Simple function to check if an ant is present in an image.
    
    Args:
        model_path: Path to trained classification model
        image_path: Path to image
        threshold: Confidence threshold
        
    Returns:
        Tuple of (is_ant: bool, confidence: float)
    """
    model = YOLO(model_path)
    results = model.predict(source=image_path, verbose=False)
    
    if len(results) > 0 and results[0].probs is not None:
        probs = results[0].probs
        top1_idx = probs.top1
        top1_conf = probs.top1conf.item()
        class_name = results[0].names[top1_idx]
        
        is_ant = (class_name.lower() == 'ant') and (top1_conf >= threshold)
        return is_ant, top1_conf
    
    return False, 0.0


def main():
    parser = argparse.ArgumentParser(description='Predict ants in images using YOLOv8')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model weights (.pt file)')
    parser.add_argument('--source', '--image', type=str, required=True,
                        dest='source',
                        help='Image path, directory, video, or webcam (0)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS (detection mode)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save results')
    parser.add_argument('--no-save', action='store_false', dest='save',
                        help='Do not save results')
    parser.add_argument('--save-crop', action='store_true',
                        help='Save cropped detections')
    parser.add_argument('--show', action='store_true',
                        help='Display results')
    parser.add_argument('--mode', type=str, choices=['classify', 'detect'], default='classify',
                        help='Prediction mode')
    
    args = parser.parse_args()
    
    if args.mode == 'classify':
        predict_classification(
            model_path=args.model,
            source=args.source,
            conf_threshold=args.conf,
            save=args.save,
            show=args.show
        )
    else:
        predict_detection(
            model_path=args.model,
            source=args.source,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            save=args.save,
            save_crop=args.save_crop,
            show=args.show
        )


if __name__ == '__main__':
    main()
