"""
Train YOLOv8 for ant detection/classification.
Supports both classification (ant vs not-ant) and fine-tuning for detection.

Usage:
    # Classification training
    python train_yolo_detector.py --data datasets/ant_classifier/dataset.yaml --epochs 50
    
    # Quick test (1 epoch)
    python train_yolo_detector.py --epochs 1 --batch 4
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def train_classifier(
    data_yaml: str,
    model_name: str = 'yolov8n-cls.pt',
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 224,
    device: str = '',
    project: str = 'runs/classify',
    name: str = 'ant_detector'
):
    """
    Train YOLOv8 classification model for ant detection.
    
    Args:
        data_yaml: Path to dataset.yaml
        model_name: Pre-trained model to use (yolov8n-cls, yolov8s-cls, yolov8m-cls)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        device: Device to use ('', 'cpu', '0', '0,1', etc.)
        project: Project directory for saving results
        name: Run name
    """
    print(f"Loading pre-trained model: {model_name}")
    model = YOLO(model_name)
    
    # Get data path from yaml
    data_path = Path(data_yaml).parent
    
    print(f"\n{'='*50}")
    print(f"Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {img_size}")
    print(f"  Device: {device or 'auto'}")
    print(f"{'='*50}\n")
    
    # Train the model
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device if device else None,
        project=project,
        name=name,
        patience=10,  # Early stopping patience
        save=True,
        plots=True,
        verbose=True
    )
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best model saved to: {project}/{name}/weights/best.pt")
    print(f"\nTo run inference:")
    print(f"  python predict_ant.py --model {project}/{name}/weights/best.pt --image <path_to_image>")
    
    return results


def train_detector(
    data_yaml: str,
    model_name: str = 'yolov8n.pt',
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = '',
    project: str = 'runs/detect',
    name: str = 'ant_detector'
):
    """
    Train YOLOv8 object detection model for ant localization.
    Requires bounding box annotations in YOLO format.
    
    Args:
        data_yaml: Path to dataset.yaml with detection annotations
        model_name: Pre-trained model (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Input image size
        device: Device to use
        project: Project directory
        name: Run name
    """
    print(f"Loading pre-trained detection model: {model_name}")
    model = YOLO(model_name)
    
    print(f"\n{'='*50}")
    print(f"Detection Training Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Data: {data_yaml}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Image Size: {img_size}")
    print(f"{'='*50}\n")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device if device else None,
        project=project,
        name=name,
        patience=20,
        save=True,
        plots=True,
        verbose=True
    )
    
    print(f"\n✓ Detection training complete!")
    print(f"✓ Best model saved to: {project}/{name}/weights/best.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 for ant detection')
    parser.add_argument('--data', type=str, default='datasets/ant_classifier/dataset.yaml',
                        help='Path to dataset.yaml')
    parser.add_argument('--model', type=str, default='yolov8n-cls.pt',
                        help='Pre-trained model name (yolov8n-cls.pt for classification)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--imgsz', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='',
                        help='Device: "", "cpu", "0", "0,1" for GPU(s)')
    parser.add_argument('--project', type=str, default='runs/classify',
                        help='Project directory for results')
    parser.add_argument('--name', type=str, default='ant_detector',
                        help='Run name')
    parser.add_argument('--mode', type=str, choices=['classify', 'detect'], default='classify',
                        help='Training mode: classify or detect')
    
    args = parser.parse_args()
    
    if args.mode == 'classify':
        train_classifier(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name
        )
    else:
        # For detection mode, use different defaults
        if args.model == 'yolov8n-cls.pt':
            args.model = 'yolov8n.pt'
        if args.imgsz == 224:
            args.imgsz = 640
        if args.project == 'runs/classify':
            args.project = 'runs/detect'
            
        train_detector(
            data_yaml=args.data,
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=args.device,
            project=args.project,
            name=args.name
        )


if __name__ == '__main__':
    main()
