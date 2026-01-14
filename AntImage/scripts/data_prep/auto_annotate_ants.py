"""
Auto-Annotation Workflow using YOLO-World

This script uses YOLO-World (zero-shot detection) to automatically generate
bounding box annotations for your ant images, which can then be used to
train a YOLO11 detection model.

Usage:
    # Generate annotations for all images in your dataset
    python auto_annotate_ants.py --input images --output datasets/ant_detection
    
    # Use custom confidence threshold
    python auto_annotate_ants.py --input images --output datasets/ant_detection --conf 0.01
    
    # Preview mode (no files saved, just counts)
    python auto_annotate_ants.py --input images --preview
"""

from ultralytics import YOLO
from pathlib import Path
import cv2
import yaml
import argparse
import random
from tqdm import tqdm
import shutil


def auto_annotate_images(
    input_dir: str,
    output_dir: str,
    conf_threshold: float = 0.01,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    class_prompts: list = None,
    preview: bool = False,
    max_images: int = None
):
    """
    Auto-annotate images using YOLO-World zero-shot detection.
    
    Args:
        input_dir: Directory containing images (can have subdirectories)
        output_dir: Output directory for YOLO format dataset
        conf_threshold: Detection confidence threshold
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set (rest goes to test)
        class_prompts: Classes to detect (default: ["ant"])
        preview: If True, only count without saving
        max_images: Maximum number of images to process (for testing)
    """
    if class_prompts is None:
        class_prompts = ["ant"]
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(input_path.rglob(f'*{ext}'))
        all_images.extend(input_path.rglob(f'*{ext.upper()}'))
    
    all_images = list(set(all_images))  # Remove duplicates
    random.shuffle(all_images)
    
    if max_images:
        all_images = all_images[:max_images]
    
    print(f"\n{'='*60}")
    print(f"YOLO-World Auto-Annotation")
    print(f"{'='*60}")
    print(f"Input directory: {input_path}")
    print(f"Found {len(all_images)} images")
    print(f"Class prompts: {class_prompts}")
    print(f"Confidence threshold: {conf_threshold}")
    
    if preview:
        print("\n[PREVIEW MODE - No files will be saved]")
    
    # Load YOLO-World model
    print("\nLoading YOLO-World model...")
    model = YOLO("yolov8l-worldv2.pt")
    model.set_classes(class_prompts)
    
    # Process images
    annotations = {}  # image_path -> list of (class_id, x_center, y_center, width, height)
    detection_count = 0
    images_with_detections = 0
    
    print("\nProcessing images...")
    for img_path in tqdm(all_images, desc="Annotating"):
        try:
            # Run detection
            results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)
            
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                if len(boxes) > 0:
                    images_with_detections += 1
                    detection_count += len(boxes)
                    
                    # Get image dimensions
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    img_h, img_w = img.shape[:2]
                    
                    # Convert to YOLO format (class_id, x_center, y_center, width, height)
                    image_annotations = []
                    for box in boxes:
                        # Get box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Convert to YOLO format (normalized)
                        x_center = ((x1 + x2) / 2) / img_w
                        y_center = ((y1 + y2) / 2) / img_h
                        width = (x2 - x1) / img_w
                        height = (y2 - y1) / img_h
                        
                        # All detections are class 0 (ant)
                        class_id = 0
                        
                        image_annotations.append((class_id, x_center, y_center, width, height))
                    
                    annotations[img_path] = image_annotations
                    
        except Exception as e:
            print(f"\nError processing {img_path}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Annotation Summary")
    print(f"{'='*60}")
    print(f"Total images processed: {len(all_images)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {detection_count}")
    print(f"Average detections per image: {detection_count/max(1, images_with_detections):.2f}")
    
    if preview:
        print("\n[PREVIEW MODE] No files saved.")
        return annotations
    
    # Split into train/val/test
    annotated_images = list(annotations.keys())
    random.shuffle(annotated_images)
    
    n_total = len(annotated_images)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_images = annotated_images[:n_train]
    val_images = annotated_images[n_train:n_train + n_val]
    test_images = annotated_images[n_train + n_val:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_images)} images")
    print(f"  Val:   {len(val_images)} images")
    print(f"  Test:  {len(test_images)} images")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Save annotations and copy images
    def save_split(images, split_name):
        for img_path in tqdm(images, desc=f"Saving {split_name}"):
            # Copy image
            dest_img = output_path / 'images' / split_name / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Save label file
            label_file = output_path / 'labels' / split_name / (img_path.stem + '.txt')
            with open(label_file, 'w') as f:
                for ann in annotations[img_path]:
                    class_id, x, y, w, h = ann
                    f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    
    print("\nSaving dataset...")
    save_split(train_images, 'train')
    save_split(val_images, 'val')
    save_split(test_images, 'test')
    
    # Create dataset.yaml
    yaml_content = {
        'path': str(output_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'ant'},
        'nc': 1
    }
    
    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Dataset saved to: {output_path}")
    print(f"✓ Config saved to: {yaml_path}")
    print(f"\nNext step: Train YOLO11 with:")
    print(f"  python train_yolo11.py --mode detect --data {yaml_path}")
    print(f"{'='*60}")
    
    return annotations


def main():
    parser = argparse.ArgumentParser(description='Auto-annotate ant images using YOLO-World')
    parser.add_argument('--input', type=str, default='images',
                        help='Input directory containing images')
    parser.add_argument('--output', type=str, default='datasets/ant_detection',
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--conf', type=float, default=0.01,
                        help='Detection confidence threshold')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Validation set ratio')
    parser.add_argument('--preview', action='store_true',
                        help='Preview mode - only count, no files saved')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum images to process (for testing)')
    parser.add_argument('--classes', type=str, nargs='+', 
                        default=['ant', 'insect'],
                        help='Class prompts for detection')
    
    args = parser.parse_args()
    
    auto_annotate_images(
        input_dir=args.input,
        output_dir=args.output,
        conf_threshold=args.conf,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        class_prompts=args.classes,
        preview=args.preview,
        max_images=args.max_images
    )


if __name__ == '__main__':
    main()
