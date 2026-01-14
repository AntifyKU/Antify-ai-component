#!/usr/bin/env python3
"""Count unique species in iNaturalist data."""

import pandas as pd
from pathlib import Path

CSV_FILE = Path(__file__).parent / "data" / "inaturalist_labels.csv"

def main():
    print("=" * 50)
    print("🐜 iNaturalist Species Count")
    print("=" * 50)
    
    df = pd.read_csv(CSV_FILE)
    
    print(f"\n📊 Total image records: {len(df)}")
    print(f"🔬 Unique species: {df['scientific_name'].nunique()}")
    print(f"📁 Unique genera: {df['genus'].nunique()}")
    
    print("\n" + "-" * 50)
    print("Top 20 species by image count:")
    print("-" * 50)
    
    species_counts = df['scientific_name'].value_counts()
    for i, (species, count) in enumerate(species_counts.head(50).items(), 1):
        print(f"{i:2}. {species}: {count} images")
    
    print("\n" + "-" * 50)
    print("All species counts:")
    print("-" * 50)
    for species, count in species_counts.items():
        print(f"  {species}: {count}")

if __name__ == "__main__":
    main()
