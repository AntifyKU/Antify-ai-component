#!/usr/bin/env python3
"""
Download ALL Thai Ant Images from GBIF
Downloads all available Thai ant specimen images from the Global Biodiversity Information Facility.
GBIF contains the same data as AntWeb but with open API access.
"""

import os
import csv
import time
import requests
from pathlib import Path
from tqdm import tqdm
from datetime import datetime


# GBIF API Configuration
GBIF_API_BASE = "https://api.gbif.org/v1"

# Output configuration
OUTPUT_DIR = Path(__file__).parent / "images"
LABELS_FILE = Path(__file__).parent / "labels.csv"

# Rate limiting
REQUEST_DELAY = 0.3  # seconds between requests (faster for GBIF)
PAGE_SIZE = 300  # GBIF max per page


def get_all_thai_ant_specimens() -> list[dict]:
    """
    Get ALL Thai ant specimens from GBIF with images.
    Uses pagination to retrieve all available records.
    
    Returns:
        List of specimen dictionaries
    """
    print("🌐 Fetching ALL Thai ant specimens from GBIF...")
    print("   (This may take a few minutes for 8000+ specimens)")
    print()
    
    url = f"{GBIF_API_BASE}/occurrence/search"
    
    specimens = []
    offset = 0
    total_count = None
    
    while True:
        params = {
            "country": "TH",  # Thailand ISO code
            "familyKey": 4342,  # Formicidae (ants)
            "hasCoordinate": True,
            "hasGeospatialIssue": False,
            "mediaType": "StillImage",  # Only records with images
            "limit": PAGE_SIZE,
            "offset": offset,
        }
        
        try:
            response = requests.get(url, params=params, timeout=60)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                end_of_records = data.get("endOfRecords", True)
                
                if total_count is None:
                    total_count = data.get("count", 0)
                    print(f"📊 Total Thai ant specimens with images: {total_count}")
                    print()
                
                if not results:
                    break
                
                for record in results:
                    # Convert GBIF format to our expected format
                    media = record.get("media", [])
                    if media:
                        specimen = {
                            "catalogNumber": record.get("catalogNumber", str(record.get("key", ""))),
                            "genus": record.get("genus", "Unknown"),
                            "species": record.get("specificEpithet", "sp."),
                            "scientificName": record.get("scientificName", "Unknown species"),
                            "country": "Thailand",
                            "locality": record.get("locality", ""),
                            "decimalLatitude": record.get("decimalLatitude"),
                            "decimalLongitude": record.get("decimalLongitude"),
                            "images": [
                                {
                                    "url": m.get("identifier", ""),
                                    "type": "specimen"
                                }
                                for m in media if m.get("type") == "StillImage"
                            ]
                        }
                        if specimen["images"]:
                            specimens.append(specimen)
                
                print(f"  Retrieved {len(specimens):,} specimens so far...")
                
                # Check if we've reached the end
                if end_of_records or len(results) < PAGE_SIZE:
                    break
                    
                offset += PAGE_SIZE
                time.sleep(REQUEST_DELAY)
            else:
                print(f"  ❌ GBIF query failed with status {response.status_code}")
                break
                
        except requests.RequestException as e:
            print(f"  ❌ Error querying GBIF: {e}")
            break
    
    print(f"\n✅ Retrieved {len(specimens):,} specimens with images")
    return specimens


def download_image(url: str, save_path: Path) -> bool:
    """
    Download an image from URL and save to disk.
    """
    try:
        response = requests.get(url, timeout=30, stream=True)
        
        if response.status_code == 200:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        else:
            return False
            
    except requests.RequestException:
        return False


def sanitize_filename(name: str) -> str:
    """Sanitize a string for use as a filename/directory name."""
    for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', ' ']:
        name = name.replace(char, '_')
    return name


def download_all_thai_ant_images(max_images_per_species: int = 0):
    """
    Main function to download ALL Thai ant images.
    
    Args:
        max_images_per_species: Maximum images per species (0 = unlimited)
    """
    print("=" * 70)
    print("🇹🇭 GBIF Thai Ant Image Downloader - COMPLETE DATASET")
    print("=" * 70)
    print()
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get ALL specimens
    specimens = get_all_thai_ant_specimens()
    
    if not specimens:
        print("❌ No specimens found. Please check your internet connection.")
        return
    
    print(f"\n📊 Processing {len(specimens):,} specimens")
    
    # Track species counts
    species_counts: dict[str, int] = {}
    labels_data = []
    downloaded_count = 0
    skipped_count = 0
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Downloading images to: {OUTPUT_DIR}")
    print()
    
    for specimen in tqdm(specimens, desc="Downloading images", unit="specimen"):
        genus = specimen.get("genus", "Unknown")
        species = specimen.get("species", "sp.")
        scientific_name = f"{genus} {species}"
        species_key = sanitize_filename(scientific_name)
        
        # Check species limit if set
        if max_images_per_species > 0:
            if species_counts.get(species_key, 0) >= max_images_per_species:
                continue
        
        # Get images for this specimen
        images = specimen.get("images", [])
        
        if not images:
            continue
        
        # Create species folder
        species_dir = OUTPUT_DIR / species_key
        
        for img_info in images:
            url = img_info.get("url", "")
            img_type = img_info.get("type", "specimen")
            
            if not url:
                continue
            
            # Generate filename
            catalog = sanitize_filename(str(specimen.get("catalogNumber", f"spec_{downloaded_count}")))
            filename = f"{catalog}_{img_type}.jpg"
            save_path = species_dir / filename
            
            # Skip if already downloaded
            if save_path.exists():
                skipped_count += 1
                continue
            
            # Download image
            if download_image(url, save_path):
                downloaded_count += 1
                species_counts[species_key] = species_counts.get(species_key, 0) + 1
                
                # Add to labels
                labels_data.append({
                    "specimen_id": specimen.get("catalogNumber", ""),
                    "genus": genus,
                    "species": species,
                    "scientific_name": scientific_name,
                    "country": specimen.get("country", "Thailand"),
                    "locality": specimen.get("locality", ""),
                    "latitude": specimen.get("decimalLatitude", ""),
                    "longitude": specimen.get("decimalLongitude", ""),
                    "image_path": str(save_path.relative_to(Path(__file__).parent)),
                    "image_type": img_type,
                })
            
            # Rate limiting
            time.sleep(REQUEST_DELAY)
    
    # Write labels CSV (append mode if exists, otherwise create)
    if labels_data:
        print(f"\n📝 Writing labels to: {LABELS_FILE}")
        
        # Check if file exists to determine write mode
        file_exists = LABELS_FILE.exists()
        mode = "a" if file_exists else "w"
        
        with open(LABELS_FILE, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=labels_data[0].keys())
            if not file_exists:
                writer.writeheader()
            writer.writerows(labels_data)
    
    # Summary
    print()
    print("=" * 70)
    print("✅ Download Complete!")
    print("=" * 70)
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"📸 New images downloaded: {downloaded_count:,}")
    print(f"⏭️  Images skipped (already exist): {skipped_count:,}")
    print(f"🐜 Species with images: {len(species_counts)}")
    print(f"📁 Images saved to: {OUTPUT_DIR}")
    print(f"📄 Labels saved to: {LABELS_FILE}")
    print()
    
    # Show species breakdown
    if species_counts:
        print("Top 20 species by image count:")
        for species, count in sorted(species_counts.items(), key=lambda x: -x[1])[:20]:
            display_name = species.replace("_", " ")
            print(f"  • {display_name}: {count:,} images")
        if len(species_counts) > 20:
            print(f"  ... and {len(species_counts) - 20} more species")


if __name__ == "__main__":
    # Configuration
    # Set to 0 for unlimited images per species
    # Set to a number (e.g., 100) to limit images per species for balanced dataset
    MAX_IMAGES_PER_SPECIES = 0  # 0 = download ALL images
    
    download_all_thai_ant_images(
        max_images_per_species=MAX_IMAGES_PER_SPECIES
    )
