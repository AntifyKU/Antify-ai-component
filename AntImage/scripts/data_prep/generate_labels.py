#!/usr/bin/env python3
"""
Generate labels from existing downloaded images.
Scans the images folder and creates a labels.csv that matches all downloaded images.
"""

import csv
import re
from pathlib import Path
from datetime import datetime


# Configuration
IMAGES_DIR = Path(__file__).parent / "images"
OUTPUT_FILE = Path(__file__).parent / "labels_synced.csv"


def extract_info_from_path(species_folder: str, filename: str) -> dict:
    """
    Extract species information from folder name and filename.
    
    Args:
        species_folder: Folder name like "Oecophylla_smaragdina"
        filename: File name like "333558933_specimen.jpg"
    
    Returns:
        Dictionary with extracted information
    """
    # Parse genus and species from folder name
    parts = species_folder.replace("_", " ").split()
    
    if len(parts) >= 2:
        genus = parts[0]
        species = parts[1] if parts[1] != "sp." else "sp."
    else:
        genus = parts[0] if parts else "Unknown"
        species = "sp."
    
    # Parse specimen ID and image type from filename
    # Format: "333558933_specimen.jpg" or "casent001_head.jpg"
    name_without_ext = filename.rsplit(".", 1)[0]
    parts = name_without_ext.rsplit("_", 1)
    
    if len(parts) >= 2:
        specimen_id = parts[0]
        image_type = parts[1]
    else:
        specimen_id = name_without_ext
        image_type = "specimen"
    
    scientific_name = f"{genus} {species}"
    
    return {
        "specimen_id": specimen_id,
        "genus": genus,
        "species": species,
        "scientific_name": scientific_name,
        "country": "Thailand",
        "image_type": image_type,
    }


def generate_labels_from_images():
    """
    Scan the images directory and generate labels.csv for all existing images.
    """
    print("=" * 60)
    print("🏷️  Label Generator from Existing Images")
    print("=" * 60)
    print()
    
    if not IMAGES_DIR.exists():
        print(f"❌ Images directory not found: {IMAGES_DIR}")
        return
    
    labels_data = []
    species_counts = {}
    
    # Scan all species folders
    species_folders = sorted([f for f in IMAGES_DIR.iterdir() if f.is_dir()])
    
    print(f"📁 Found {len(species_folders)} species folders")
    print()
    
    for species_folder in species_folders:
        species_name = species_folder.name
        
        # Get all image files in this folder
        image_files = sorted([
            f for f in species_folder.iterdir() 
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
        ])
        
        species_counts[species_name] = len(image_files)
        
        for image_file in image_files:
            # Extract info from path
            info = extract_info_from_path(species_name, image_file.name)
            
            # Add image path relative to parent of images folder
            info["image_path"] = str(image_file.relative_to(Path(__file__).parent))
            
            labels_data.append(info)
    
    # Write labels CSV
    if labels_data:
        print(f"📝 Writing labels to: {OUTPUT_FILE}")
        
        fieldnames = ["specimen_id", "genus", "species", "scientific_name", "country", "image_path", "image_type"]
        
        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(labels_data)
        
        # Summary
        print()
        print("=" * 60)
        print("✅ Labels Generated!")
        print("=" * 60)
        print(f"📸 Total images labeled: {len(labels_data)}")
        print(f"🐜 Total species: {len(species_counts)}")
        print(f"📄 Output file: {OUTPUT_FILE}")
        print()
        
        # Show species breakdown
        print("Top 15 species by image count:")
        for species, count in sorted(species_counts.items(), key=lambda x: -x[1])[:15]:
            # Convert underscore back to space for display
            display_name = species.replace("_", " ")
            print(f"  • {display_name}: {count} images")
        if len(species_counts) > 15:
            print(f"  ... and {len(species_counts) - 15} more species")
    else:
        print("❌ No images found in the images folder")


if __name__ == "__main__":
    generate_labels_from_images()
