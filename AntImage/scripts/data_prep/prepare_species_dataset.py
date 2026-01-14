"""
Prepare dataset for YOLOv8/YOLO11 species classification.
Organizes ant images into train/val/test splits with SPECIES SUBFOLDERS.

Usage:
    python prepare_species_dataset.py [--output datasets/ant_species] [--dry-run]
    
Structure created:
    datasets/ant_species/
    ├── train/
    │   ├── Oecophylla_smaragdina/
    │   │   ├── img1.jpg
    │   │   └── img2.jpg
    │   ├── Camponotus_festinus/
    │   │   └── ...
    │   └── ...
    ├── val/
    │   ├── Oecophylla_smaragdina/
    │   └── ...
    └── test/
        └── ...
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import csv
from collections import defaultdict


def prepare_species_dataset(
    images_dir: str,
    labels_csv: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    min_samples: int = 5,  # Minimum samples per species
    seed: int = 42,
    dry_run: bool = False
):
    """
    Prepare dataset for YOLO species classification.
    Organizes images into species-based subfolders.
    """
    random.seed(seed)
    
    output_path = Path(output_dir)
    images_path = Path(images_dir)
    
    # Collect images grouped by species
    species_images = defaultdict(list)
    
    print(f"Reading labels from {labels_csv}...")
    with open(labels_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = images_path.parent / row['image_path'].replace('\\', '/')
            if img_path.exists():
                species_name = row['scientific_name'].replace(' ', '_')
                species_images[species_name].append(img_path)
    
    print(f"Found {sum(len(v) for v in species_images.values())} valid images")
    print(f"Found {len(species_images)} unique species")
    
    # Filter species with minimum samples
    filtered_species = {k: v for k, v in species_images.items() if len(v) >= min_samples}
    excluded = set(species_images.keys()) - set(filtered_species.keys())
    
    if excluded:
        print(f"\nExcluded {len(excluded)} species with < {min_samples} samples:")
        for sp in list(excluded)[:5]:
            print(f"  - {sp}: {len(species_images[sp])} images")
        if len(excluded) > 5:
            print(f"  ... and {len(excluded) - 5} more")
    
    species_images = filtered_species
    print(f"\nUsing {len(species_images)} species with >= {min_samples} samples")
    
    # Show top species
    print("\nTop 10 species by image count:")
    sorted_species = sorted(species_images.items(), key=lambda x: len(x[1]), reverse=True)
    for species, images in sorted_species[:10]:
        print(f"  {species}: {len(images)} images")
    
    if dry_run:
        total = sum(len(v) for v in species_images.values())
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)
        n_test = total - n_train - n_val
        print(f"\n[DRY RUN] Would create:")
        print(f"  Train: ~{n_train} images")
        print(f"  Val:   ~{n_val} images")
        print(f"  Test:  ~{n_test} images")
        print(f"  Classes: {len(species_images)}")
        print("\n[DRY RUN] No files were copied.")
        return
    
    # Create directories and copy files
    stats = {'train': 0, 'val': 0, 'test': 0}
    
    for species_name, images in tqdm(species_images.items(), desc="Processing species"):
        # Shuffle images for this species
        random.shuffle(images)
        
        # Split for this species
        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)
        
        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]
        
        splits = {
            'train': train_imgs,
            'val': val_imgs,
            'test': test_imgs
        }
        
        for split_name, split_imgs in splits.items():
            if not split_imgs:
                continue
                
            split_dir = output_path / split_name / species_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_imgs:
                dest = split_dir / img_path.name
                # Handle duplicates
                if dest.exists():
                    dest = split_dir / f"{img_path.stem}_{random.randint(1000,9999)}{img_path.suffix}"
                shutil.copy2(img_path, dest)
                stats[split_name] += 1
    
    print(f"\n{'='*50}")
    print(f"Dataset Statistics:")
    print(f"{'='*50}")
    print(f"  Train: {stats['train']} images")
    print(f"  Val:   {stats['val']} images")
    print(f"  Test:  {stats['test']} images")
    print(f"  Total: {sum(stats.values())} images")
    print(f"  Classes: {len(species_images)}")
    
    # Create dataset.yaml for YOLOv8/YOLO11
    names_dict = {i: name for i, name in enumerate(sorted(species_images.keys()))}
    
    yaml_content = f"""# YOLOv8/YOLO11 Species Classification Dataset
# Auto-generated by prepare_species_dataset.py

path: {output_path.absolute()}
train: train
val: val
test: test

# Number of classes
nc: {len(species_images)}

# Class names
names:
"""
    for idx, name in names_dict.items():
        yaml_content += f"  {idx}: {name}\n"
    
    yaml_path = output_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Dataset prepared at: {output_path.absolute()}")
    print(f"✓ Config saved to: {yaml_path}")
    print(f"\nNext step: Train species classifier with:")
    print(f"  python train_yolo11.py --mode classify --data-dir {output_path} --epochs 100")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for species classification')
    parser.add_argument('--images', type=str, default='images',
                        help='Path to images directory')
    parser.add_argument('--labels', type=str, default='labels.csv',
                        help='Path to labels CSV file')
    parser.add_argument('--output', type=str, default='datasets/ant_species',
                        help='Output directory for prepared dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of training data (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Ratio of validation data (default: 0.2)')
    parser.add_argument('--min-samples', type=int, default=5,
                        help='Minimum samples per species (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without copying files')
    
    args = parser.parse_args()
    
    prepare_species_dataset(
        images_dir=args.images,
        labels_csv=args.labels,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_samples=args.min_samples,
        seed=args.seed,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
