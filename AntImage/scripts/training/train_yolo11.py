"""
YOLO11 Training Script for Ant Detection/Classification

Usage:
  For Classification (species ID):
    python train_yolo11.py --mode classify --epochs 150
    
  For Detection (requires bounding box annotations):
    python train_yolo11.py --mode detect --epochs 150 --data antify_data.yaml
"""

from ultralytics import YOLO
import argparse
from pathlib import Path


def train_classifier(
    data_path: str = "datasets/ant_classifier",
    epochs: int = 100,
    batch_size: int = 8,
    img_size: int = 640,
    model_name: str = "yolo11m-cls.pt",
    project: str = "runs/classify",
    name: str = "ant_species_v1"
):
    """
    Train YOLO11 for ant species classification.
    Uses your existing labeled images (no bounding boxes needed).
    """
    print("="*60)
    print("YOLO11 Species Classification Training")
    print("="*60)
    
    # Load YOLO11 classification model
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=data_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=20,           # Early stopping
        lr0=0.01,              # Initial learning rate
        lrf=0.01,              # Final learning rate fraction
        momentum=0.937,
        weight_decay=0.0005,
        augment=True,
        project=project,
        name=name,
        verbose=True,
        plots=True
    )
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best model saved to: {project}/{name}/weights/best.pt")
    
    return results


def train_detector(
    data_yaml: str = "antify_data.yaml",
    epochs: int = 150,
    batch_size: int = 8,              # Reduced for 8GB GPU
    img_size: int = 640,              # Reduced for 8GB GPU (was 640)
    model_name: str = "yolo11s.pt",   # Use Small model for 8GB GPU
    project: str = "runs/detect",
    name: str = "antify_production_v1"
):
    """
    Train YOLO11 for ant detection (requires bounding box annotations).
    Your antify_data.yaml should point to annotated data.
    """
    print("="*60)
    print("YOLO11 Object Detection Training")
    print("="*60)
    
    # Check if data file exists
    if not Path(data_yaml).exists():
        print(f"\n❌ Error: {data_yaml} not found!")
        print("\nTo create detection annotations, you need:")
        print("  1. Use Roboflow, Label Studio, or CVAT to annotate bounding boxes")
        print("  2. Export in YOLO format")
        print("  3. Create antify_data.yaml pointing to the annotated data")
        print("\nExample antify_data.yaml:")
        print("""
path: ./datasets/antify
train: images/train
val: images/val
test: images/test

names:
  0: ant
""")
        return None
    
    # Load YOLO11 detection model
    model = YOLO(model_name)
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=20,           # Early stopping to prevent overfitting
        lr0=0.01,              # Initial learning rate
        lrf=0.01,              # Final learning rate fraction
        momentum=0.937,        # SGD momentum
        weight_decay=0.0005,   # Regularization
        augment=True,          # Enable default augmentations
        mosaic=1.0,            # Mosaic augmentation (critical for small objects)
        mixup=0.1,             # Light MixUp
        project=project,
        name=name,
        verbose=True,
        plots=True
    )
    
    print(f"\n✓ Training complete!")
    print(f"✓ Best model saved to: {project}/{name}/weights/best.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 for ant detection/classification')
    parser.add_argument('--mode', type=str, choices=['detect', 'classify'], default='classify',
                        help='Training mode: detect (requires annotations) or classify (species ID)')
    parser.add_argument('--data', type=str, default='antify_data.yaml',
                        help='Path to data.yaml for detection mode')
    parser.add_argument('--data-dir', type=str, default='datasets/ant_classifier',
                        help='Path to dataset directory for classification mode')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Input image size')
    parser.add_argument('--model', type=str, default='', help='Model name (auto-selected if empty)')
    parser.add_argument('--name', type=str, default='', help='Run name')
    
    args = parser.parse_args()
    
    if args.mode == 'classify':
        model_name = args.model if args.model else 'yolo11m-cls.pt'
        run_name = args.name if args.name else 'ant_species_v1'
        train_classifier(
            data_path=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            model_name=model_name,
            name=run_name
        )
    else:
        model_name = args.model if args.model else 'yolo11s.pt'  # Small model for 8GB GPU
        run_name = args.name if args.name else 'antify_production_v1'
        train_detector(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            model_name=model_name,
            name=run_name
        )


if __name__ == '__main__':
    main()
