"""
YOLO11 Inference Script for Ant Detection

Usage:
    python scripts/inference/test_yolo11.py --source datasets/ant_detection/images/test
    python scripts/inference/test_yolo11.py --source path/to/image.jpg
    python scripts/inference/test_yolo11.py --source path/to/folder
"""

from ultralytics import YOLO
import argparse
from pathlib import Path
import cv2
import os


def run_inference(
    model_path: str = "runs/detect/runs/detect/antify_production_v14/weights/best.pt",
    source: str = "datasets/ant_detection/images/test",
    conf_threshold: float = 0.25,
    save_results: bool = True,
    show_results: bool = False,
    save_txt: bool = False,
    project: str = "runs/inference",
    name: str = "test_results"
):
    """
    Run inference on images using trained YOLO11 model.
    """
    print("=" * 60)
    print("YOLO11 Ant Detection Inference")
    print("=" * 60)
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ Error: Model not found at {model_path}")
        print("\nAvailable models in runs/detect:")
        detect_dir = Path("runs/detect")
        if detect_dir.exists():
            for run_dir in detect_dir.rglob("best.pt"):
                print(f"  - {run_dir}")
        return None
    
    # Check if source exists
    if not Path(source).exists():
        print(f"\n❌ Error: Source not found at {source}")
        return None
    
    print(f"\n📦 Loading model: {model_path}")
    model = YOLO(model_path)
    
    print(f"📁 Source: {source}")
    print(f"🎯 Confidence threshold: {conf_threshold}")
    
    # Run inference
    results = model.predict(
        source=source,
        conf=conf_threshold,
        save=save_results,
        save_txt=save_txt,
        show=show_results,
        project=project,
        name=name,
        exist_ok=True
    )
    
    # Print summary
    total_images = len(results)
    total_detections = sum(len(r.boxes) for r in results)
    
    print(f"\n✓ Inference complete!")
    print(f"  - Images processed: {total_images}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Avg detections/image: {total_detections / total_images:.2f}")
    
    if save_results:
        print(f"  - Results saved to: {project}/{name}")
    
    return results


def test_single_image(model_path: str, image_path: str, conf_threshold: float = 0.25):
    """
    Test on a single image and display results.
    """
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf_threshold)
    
    # Get the result
    result = results[0]
    
    # Print detections
    print(f"\n🖼️  Image: {image_path}")
    print(f"   Detections: {len(result.boxes)}")
    
    for i, box in enumerate(result.boxes):
        conf = box.conf[0].item()
        cls = int(box.cls[0].item())
        xyxy = box.xyxy[0].tolist()
        print(f"   [{i+1}] Class: {result.names[cls]}, Confidence: {conf:.3f}, Box: {xyxy}")
    
    # Save annotated image
    annotated = result.plot()
    output_path = Path(image_path).stem + "_detected.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"\n✓ Saved annotated image to: {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Run YOLO11 inference for ant detection')
    parser.add_argument('--model', type=str, 
                        default='runs/detect/runs/detect/antify_production_v14/weights/best.pt',
                        help='Path to trained model weights')
    parser.add_argument('--source', type=str, 
                        default='datasets/ant_detection/images/test',
                        help='Path to image, folder, or video')
    parser.add_argument('--conf', type=float, default=0.25, 
                        help='Confidence threshold (0-1)')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save results with bounding boxes')
    parser.add_argument('--show', action='store_true',
                        help='Display results in window')
    parser.add_argument('--save-txt', action='store_true',
                        help='Save results as txt files')
    parser.add_argument('--name', type=str, default='test_results',
                        help='Name for results folder')
    
    args = parser.parse_args()
    
    run_inference(
        model_path=args.model,
        source=args.source,
        conf_threshold=args.conf,
        save_results=args.save,
        show_results=args.show,
        save_txt=args.save_txt,
        name=args.name
    )


if __name__ == '__main__':
    main()
