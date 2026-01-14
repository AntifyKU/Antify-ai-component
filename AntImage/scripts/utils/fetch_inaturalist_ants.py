#!/usr/bin/env python3
"""
Fetch iNaturalist Ant Images for Thailand
Downloads ant observations and images from iNaturalist for Thailand.
"""

import os
import csv
import time
import requests
from pathlib import Path
from tqdm import tqdm

# Configuration
INAT_API_BASE = "https://api.inaturalist.org/v1"
PLACE_ID = 6967  # Thailand
TAXON_ID = 47336  # Formicidae (Ants)
# QUALITY_GRADE = "research" # Optional: restrict to research grade
PAGE_SIZE = 200

# Output Paths
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "inaturalist"
DATA_DIR = BASE_DIR / "data"
LABELS_FILE = DATA_DIR / "inaturalist_labels.csv"

def sanitize_filename(name):
    """Sanitize a string for use as a filename."""
    return "".join(c if c.isalnum() or c in ('-','_') else '_' for c in name)

def get_inaturalist_observations(max_results=None):
    """Fetch observations from iNaturalist API using cursor-based pagination.
    
    Uses id_above parameter for unlimited results (bypasses 10,000 page limit).
    """
    print("🌐 Fetching iNaturalist observations...")
    
    observations = []
    last_id = 0  # Start from beginning
    batch_num = 0
    
    while True:
        params = {
            "place_id": PLACE_ID,
            "taxon_id": TAXON_ID,
            "quality_grade": "research,needs_id",  # reliable data
            "per_page": PAGE_SIZE,
            "photos": "true",
            "geo": "true",
            "order_by": "id",
            "order": "asc",
            "id_above": last_id  # Cursor-based pagination
        }
        
        try:
            response = requests.get(f"{INAT_API_BASE}/observations", params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                if not results:
                    print("  ✓ No more results available.")
                    break
                
                observations.extend(results)
                last_id = results[-1]["id"]  # Update cursor to last ID
                batch_num += 1
                
                print(f"  Batch {batch_num}: fetched {len(results)} (IDs up to {last_id}). Total: {len(observations)}")
                
                if max_results and len(observations) >= max_results:
                    observations = observations[:max_results]
                    print(f"  ✓ Reached max_results limit: {max_results}")
                    break
                
                time.sleep(1.0)  # Respect rate limits
            elif response.status_code == 429:
                print("  ⚠️ Rate limited. Waiting 60 seconds...")
                time.sleep(60)
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
                break
        except Exception as e:
            print(f"❌ Exception: {e}")
            print("  Retrying in 10 seconds...")
            time.sleep(10)
            continue
            
    return observations

def download_image(url, save_path):
    """Download image to path."""
    try:
        if save_path.exists():
            return True # Skip
            
        headers = {
            "User-Agent": "AntImageBot/1.0 (Contact: mail@example.com)"
        }
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except:
        pass
    return False

def main():
    print("="*60)
    print("🐜 iNaturalist Thai Ant Photo Downloader")
    print("="*60)
    
    # Setup directories
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch Data
    # Now using id_above cursor pagination for unlimited results.
    # Set max_results=None for all available data, or a number to limit.
    
    obs_list = get_inaturalist_observations(max_results=None)  # Fetch ALL available
    
    if not obs_list:
        print("No observations found.")
        return

    print(f"\nProcessing {len(obs_list)} observations...")
    
    downloaded_count = 0
    
    # Initialize CSV
    file_exists = LABELS_FILE.exists()
    
    if not file_exists:
        with open(LABELS_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "source", "id", "scientific_name", "genus", "species",
                "lat", "lon", "locality", "image_path", "image_url", "date_observed"
            ])
            writer.writeheader()
    
    for obs in tqdm(obs_list, desc="Downloading"):
        # Extract metadata
        try:
            obs_id = obs['id']
            taxon = obs.get('taxon', {})
            if not taxon: continue
            
            scientific_name = taxon.get('name', 'Unknown')
            # rank = taxon.get('rank')
            
            # Location
            location = obs.get('location') # "lat,lon" string
            lat, lon = (None, None)
            if location:
                try:
                    lat, lon = location.split(',')
                except:
                    pass
            
            place_guess = obs.get('place_guess', '')
            
            # Photos
            photos = obs.get('photos', [])
            if not photos: continue
            
            # Prepare directory
            safe_name = sanitize_filename(scientific_name)
            species_dir = IMAGES_DIR / safe_name
            species_dir.mkdir(exist_ok=True)
            
            for i, photo in enumerate(photos):
                url = photo.get('url')
                # Get large version if possible
                url = url.replace("square.jpg", "large.jpg") if url else None
                
                if not url: continue
                
                filename = f"{obs_id}_{i}.jpg"
                save_path = species_dir / filename
                
                if download_image(url, save_path):
                    downloaded_count += 1
                    
                    row = {
                        "source": "iNaturalist",
                        "id": obs_id,
                        "scientific_name": scientific_name,
                        "genus": scientific_name.split()[0] if ' ' in scientific_name else scientific_name,
                        "species": scientific_name, # Simplified
                        "lat": lat,
                        "lon": lon,
                        "locality": place_guess,
                        "image_path": str(save_path.relative_to(BASE_DIR)),
                        "image_url": url,
                        "date_observed": obs.get('observed_on')
                    }
                    
                    # Append to CSV
                    with open(LABELS_FILE, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=row.keys())
                        writer.writerow(row)
                    
        except Exception as e:
            # print(f"Skipping obs {obs.get('id')}: {e}")
            continue
            
    print(f"\n✅ Done! Downloaded {downloaded_count} images.")
    print(f"Labels saved to {LABELS_FILE}")

if __name__ == "__main__":
    main()
