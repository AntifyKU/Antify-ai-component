import csv
import shutil
from pathlib import Path

def remove_duplicates(file_path):
    print(f"Propcessing {file_path}...")
    
    if not file_path.exists():
        print("❌ File not found.")
        return

    # Backup original
    backup_path = file_path.with_suffix('.csv.bak')
    shutil.copy2(file_path, backup_path)
    print(f"📦 Backup created at {backup_path}")

    unique_rows = []
    seen_ids = set()
    initial_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            initial_count += 1
            # distinct based on 'id' and 'image_url' to ensure we don't drop different images for same observation
            # But let's check for exact duplicates first? 
            # The user asked for "duplicate row". 
            # Looking at the data, it seems like copy-paste errors.
            # Using 'image_path' should be unique enough for this dataset.
            
            identifier = row.get('image_path')
            
            if identifier not in seen_ids:
                seen_ids.add(identifier)
                unique_rows.append(row)
                
    final_count = len(unique_rows)
    removed_count = initial_count - final_count
    
    if removed_count > 0:
        print(f"🧹 Removing {removed_count} duplicate rows...")
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unique_rows)
        print("✅ Done!")
    else:
        print("✨ No duplicates found.")
        
    print(f"Original: {initial_count} rows")
    print(f"Final:    {final_count} rows")

if __name__ == "__main__":
    target_file = Path(__file__).parent / "data" / "inaturalist_labels.csv"
    remove_duplicates(target_file)
