#!/usr/bin/env python3
"""
Fetch Top 50 Ant Species Found in Thailand - Count ALL Images Worldwide
Gets species found in Thailand, then counts their total images globally in GBIF.
"""

import sys
import io

# Fix Windows console encoding for emojis
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
import time
from tabulate import tabulate
from tqdm import tqdm

# GBIF API Configuration
GBIF_API_BASE = "https://api.gbif.org/v1"
FORMICIDAE_KEY = 4342  # Formicidae (Ants) taxon key
THAILAND_KEY = "TH"    # Thailand ISO country code


def get_thai_ant_species(limit=50):
    """
    Fetch the top ant species found in Thailand by occurrence count.
    """
    print("🌐 Fetching top ant species found in Thailand...")
    
    params = {
        "familyKey": FORMICIDAE_KEY,
        "country": THAILAND_KEY,
        "facet": "speciesKey",
        "facetLimit": limit,
        "limit": 0,
    }
    
    try:
        response = requests.get(
            f"{GBIF_API_BASE}/occurrence/search",
            params=params,
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            facets = data.get("facets", [])
            
            for facet in facets:
                if facet.get("field") == "SPECIES_KEY":
                    return facet.get("counts", [])
            return []
        else:
            print(f"❌ Error: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return []


def get_species_details(species_key):
    """Get species name and details from GBIF."""
    try:
        response = requests.get(
            f"{GBIF_API_BASE}/species/{species_key}",
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return {}


def count_all_images_worldwide(species_key):
    """
    Count ALL images available for a species WORLDWIDE (not just Thailand).
    """
    params = {
        "speciesKey": species_key,
        "mediaType": "StillImage",
        "limit": 0,
    }
    
    try:
        response = requests.get(
            f"{GBIF_API_BASE}/occurrence/search",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get("count", 0)
    except:
        pass
    return 0


def count_thai_occurrences(species_key):
    """Count occurrences in Thailand only."""
    params = {
        "speciesKey": species_key,
        "country": THAILAND_KEY,
        "limit": 0,
    }
    
    try:
        response = requests.get(
            f"{GBIF_API_BASE}/occurrence/search",
            params=params,
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("count", 0)
    except:
        pass
    return 0


def format_number(num):
    """Format numbers with commas."""
    return f"{num:,}"


def main():
    print("=" * 95)
    print("🐜 GBIF: Top 50 Thai Ants - Counting ALL Images Worldwide")
    print("=" * 95)
    
    species_data = get_thai_ant_species(limit=50)
    
    if not species_data:
        print("No data retrieved.")
        return
    
    print(f"\n📊 Found {len(species_data)} species in Thailand, counting worldwide images...\n")
    
    table_data = []
    total_thai_occ = 0
    total_worldwide_images = 0
    
    for item in tqdm(species_data, desc="Counting images"):
        species_key = item.get("name")
        thai_count = item.get("count", 0)
        total_thai_occ += thai_count
        
        # Get species details
        details = get_species_details(species_key)
        scientific_name = details.get("species", details.get("canonicalName", "Unknown"))
        genus = details.get("genus", "-")
        
        # Count ALL images worldwide
        worldwide_images = count_all_images_worldwide(species_key)
        total_worldwide_images += worldwide_images
        
        table_data.append({
            "scientific_name": scientific_name,
            "genus": genus,
            "thai_occurrences": thai_count,
            "worldwide_images": worldwide_images,
            "species_key": species_key
        })
        
        time.sleep(0.2)
    
    # Sort by worldwide images (most images first)
    table_data.sort(key=lambda x: x["worldwide_images"], reverse=True)
    
    # Display
    print("\n")
    headers = ["#", "Scientific Name", "Genus", "Thai Occ.", "Worldwide Images", "Species Key"]
    rows = []
    for i, row in enumerate(table_data, 1):
        rows.append([
            i,
            row["scientific_name"],
            row["genus"],
            format_number(row["thai_occurrences"]),
            format_number(row["worldwide_images"]),
            row["species_key"]
        ])
    
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    
    print("\n" + "-" * 95)
    print(f"📍 Total Thai occurrences: {format_number(total_thai_occ)}")
    print(f"📷 Total worldwide images (all 50 species): {format_number(total_worldwide_images)}")
    print("-" * 95)
    
    # Top genera
    print("\n🔬 Top Genera (by worldwide images):")
    genus_images = {}
    genus_species = {}
    for row in table_data:
        genus = row["genus"]
        if genus and genus != "-":
            genus_images[genus] = genus_images.get(genus, 0) + row["worldwide_images"]
            genus_species[genus] = genus_species.get(genus, 0) + 1
    
    sorted_genera = sorted(genus_images.items(), key=lambda x: x[1], reverse=True)[:10]
    for genus, images in sorted_genera:
        print(f"   {genus}: {genus_species[genus]} species ({format_number(images)} worldwide images)")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
