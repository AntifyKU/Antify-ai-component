import csv
import os
from pathlib import Path

def normalize_path(p):
    return str(p).replace('\\', '/').strip()

def main():
    base_dir = Path(__file__).parent
    # The actual folder on disk (as verified in recent steps)
    images_dir = base_dir / "inaturalist" 
    csv_path = base_dir / "data" / "inaturalist_labels.csv"

    print(f"Checking consistency...")
    print(f"Images Directory: {images_dir}")
    print(f"CSV File: {csv_path}")

    # 1. Load CSV Data
    csv_files = {}
    if not csv_path.exists():
        print("❌ CSV file not found!")
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # CSV path might be 'images\inaturalist\Species\file.jpg'
            # We want to match it to 'inaturalist/Species/file.jpg'
            orig_path = row.get('image_path', '')
            
            # Normalize to check against current disk structure
            # If CSV has 'images\inaturalist', map it to 'inaturalist'
            rel_path = orig_path.replace('\\', '/')
            if rel_path.startswith('images/inaturalist/'):
                rel_path = rel_path.replace('images/inaturalist/', 'inaturalist/')
            elif rel_path.startswith('inaturalist/'):
                pass # already correct prefix
            else:
                # Fallback: maybe just 'Species/file.jpg'? 
                # Let's assume the CSV structure is consistent with 'folder/file' relative to root
                pass

            # We'll key by the ID or the filename to be safe
            fname = os.path.basename(rel_path)
            csv_files[fname] = {
                'row_idx': i + 2, # 1-based + header
                'csv_path': orig_path,
                'expected_rel_path': rel_path 
            }

    print(f"Found {len(csv_files)} records in CSV.")

    # 2. Scan Disk
    disk_files = []
    if images_dir.exists():
        for root, _, files in os.walk(images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Create relative path from base_dir to match our normalized CSV path
                    # disk file: d:\Vscode\AntImage\inaturalist\Species\file.jpg
                    # rel path: inaturalist\Species\file.jpg
                    abs_path = Path(root) / file
                    try:
                        rel = abs_path.relative_to(base_dir)
                        disk_files.append(str(rel).replace('\\', '/'))
                    except ValueError:
                        pass
    else:
        print("❌ Images directory not found on disk!")
        return

    print(f"Found {len(disk_files)} images on disk.")
    
    # 3. Compare
    disk_dict = {os.path.basename(p): p for p in disk_files}
    
    missing_on_disk = []
    extra_on_disk = []
    
    # Check CSV -> Disk
    for fname, info in csv_files.items():
        if fname not in disk_dict:
            missing_on_disk.append(info)
            
    # Check Disk -> CSV
    for fname, path in disk_dict.items():
        if fname not in csv_files:
            extra_on_disk.append(path)

    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    
    if not missing_on_disk and not extra_on_disk:
        print("✅ SUCCESS: Data is perfectly consistent!")
    else:
        if missing_on_disk:
            print(f"\n❌ {len(missing_on_disk)} files in CSV but MISSING on disk:")
            for item in missing_on_disk[:10]:
                print(f"  Line {item['row_idx']}: {item['csv_path']}")
            if len(missing_on_disk) > 10:
                print(f"  ... and {len(missing_on_disk) - 10} more.")

        if extra_on_disk:
            print(f"\n⚠️ {len(extra_on_disk)} files on disk but NOT in CSV:")
            for path in extra_on_disk[:10]:
                print(f"  {path}")
            if len(extra_on_disk) > 10:
                print(f"  ... and {len(extra_on_disk) - 10} more.")

if __name__ == "__main__":
    main()
