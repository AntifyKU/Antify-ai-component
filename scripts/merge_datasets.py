import pandas as pd

def merge_datasets():
    # Load primary dataset
    specimen_df = pd.read_csv('labels_synced.csv')
    print(f"Specimen images: {len(specimen_df)}")
    
    # Load iNaturalist dataset
    inat_df = pd.read_csv('data/inaturalist_labels.csv')
    print(f"iNaturalist images: {len(inat_df)}")
    
    # Standardize columns
    # We need: image_path, scientific_name
    # Optional: image_type
    
    # Specimen DF already has these.
    # iNat DF has these too.
    
    # Select relevant columns
    cols = ['scientific_name', 'image_path']
    
    df1 = specimen_df[cols].copy()
    df1['source'] = 'specimen'
    
    df2 = inat_df[cols].copy()
    df2['source'] = 'inaturalist'
    
    # Merge
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    output_path = 'combined_labels.csv'
    combined_df.to_csv(output_path, index=False)
    
    print(f"Merged dataset saved to {output_path}")
    print(f"Total images: {len(combined_df)}")
    print(combined_df['source'].value_counts())

if __name__ == "__main__":
    merge_datasets()
