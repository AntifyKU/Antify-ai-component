#!/usr/bin/env python3
"""
Fetch Top 50 Ant Species in Thailand from iNaturalist by Image Count
Queries the iNaturalist API to find the most photographed ant species in Thailand.
Uses sampling to estimate photo counts quickly.
"""

import sys
import io

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
import time
from tabulate import tabulate
from tqdm import tqdm

# iNaturalist API Configuration
INAT_API_BASE = "https://api.inaturalist.org/v1"
TAXON_ID = 47336  # Formicidae (Ants)
PLACE_ID = 6967   # Thailand


def get_top_ant_species(limit=50):
    """
    Fetch the top ant species by observation count from iNaturalist.
    """
    print("🌐 Fetching top ant species from iNaturalist...")
    
    params = {
        "taxon_id": TAXON_ID,
        "place_id": PLACE_ID,
        "rank": "species",
        "per_page": limit,
        "quality_grade": "research,needs_id",
    }
    
    try:
        response = requests.get(
            f"{INAT_API_BASE}/observations/species_counts",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return []
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []


def estimate_photo_count(taxon_id, obs_count):
    """
    Estimate photo count by sampling a batch of observations.
    Takes the average photos per observation and multiplies by total.
    """
    sample_size = min(200, obs_count)  # Sample up to 200 observations
    
    params = {
        "taxon_id": taxon_id,
        "place_id": PLACE_ID,
        "quality_grade": "research,needs_id",
        "photos": "true",
        "per_page": sample_size,
    }
    
    try:
        response = requests.get(
            f"{INAT_API_BASE}/observations",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            
            if results:
                total_photos_in_sample = sum(len(obs.get("photos", [])) for obs in results)
                avg_photos = total_photos_in_sample / len(results)
                estimated_total = int(obs_count * avg_photos)
                return estimated_total, total_photos_in_sample if len(results) == obs_count else estimated_total
            
    except:
        pass
    
    return obs_count, obs_count  # Fallback: 1 photo per observation


def format_number(num):
    """Format large numbers with commas."""
    return f"{num:,}"


def main():
    print("=" * 85)
    print("🐜 iNaturalist Top 50 Ant Species in THAILAND (by Estimated Image Count)")
    print("=" * 85)
    
    species_data = get_top_ant_species(limit=50)
    
    if not species_data:
        print("No data retrieved.")
        return
    
    print(f"\n📊 Found {len(species_data)} species, estimating image counts...\n")
    
    # Prepare table data with photo counts
    table_data = []
    total_images = 0
    
    for item in tqdm(species_data, desc="Estimating images"):
        taxon = item.get("taxon", {})
        obs_count = item.get("count", 0)
        taxon_id = taxon.get("id")
        
        scientific_name = taxon.get("name", "Unknown")
        common_name = taxon.get("preferred_common_name", "-")
        
        # Get genus from scientific name
        genus = scientific_name.split()[0] if " " in scientific_name else scientific_name
        
        # Estimate photo count
        estimated_photos, _ = estimate_photo_count(taxon_id, obs_count)
        total_images += estimated_photos
        
        table_data.append({
            "scientific_name": scientific_name,
            "common_name": common_name[:25] if common_name else "-",
            "genus": genus,
            "obs_count": obs_count,
            "photo_count": estimated_photos,
            "taxon_id": taxon_id
        })
        
        time.sleep(0.3)  # Be nice to the API
    
    # Sort by photo count (descending)
    table_data.sort(key=lambda x: x["photo_count"], reverse=True)
    
    # Display results
    print("\n")
    headers = ["#", "Scientific Name", "Common Name", "Genus", "Obs", "Est. Images", "Taxon ID"]
    rows = []
    for i, row in enumerate(table_data, 1):
        rows.append([
            i,
            row["scientific_name"],
            row["common_name"],
            row["genus"],
            format_number(row["obs_count"]),
            format_number(row["photo_count"]),
            row["taxon_id"]
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    
    print("\n" + "-" * 85)
    print(f"📈 Total observations: {format_number(sum(r['obs_count'] for r in table_data))}")
    print(f"📷 Total estimated images: {format_number(total_images)}")
    print("-" * 85)
    
    # Top 10 genera by species count
    print("\n🔬 Top Genera (by species in top 50):")
    genus_counts = {}
    genus_images = {}
    for row in table_data:
        genus = row["genus"]
        genus_counts[genus] = genus_counts.get(genus, 0) + 1
        genus_images[genus] = genus_images.get(genus, 0) + row["photo_count"]
    
    sorted_genera = sorted(genus_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for genus, count in sorted_genera:
        print(f"   {genus}: {count} species ({format_number(genus_images[genus])} est. images)")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
