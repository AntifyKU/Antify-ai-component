import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tqdm import tqdm
from PIL import Image

# Import from existing scripts
from train_bioclip import AntDataset
from bioclip_model import BioCLIPClassifier

def visualize(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Data
    print(f"Loading labels from {args.csv_file}...")
    df = pd.read_csv(args.csv_file)
    
    # Create class mapping using the *Model's* class list if possible, otherwise dataset's
    # But for confusion matrix, we need to match the model's output classes.
    
    # Load Model to get classes
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'classes' in checkpoint:
        classes = checkpoint['classes']
    else:
        # Fallback to loading from classes.txt
        model_dir = os.path.dirname(args.model_path)
        class_file = os.path.join(model_dir, "classes.txt")
        if os.path.exists(class_file):
            with open(class_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
        else:
            # Fallback to dataset classes (risky if mismatch)
            classes = sorted(df['scientific_name'].unique())
            
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    print(f"Model has {len(classes)} classes.")

    # Filter dataset to only include classes that are in the model
    # (Important for Zero-Shot model which has 60 classes vs dataset's 270)
    df_filtered = df[df['scientific_name'].isin(classes)].copy()
    print(f"Filtered dataset from {len(df)} to {len(df_filtered)} samples matching model classes.")
    
    # Use a subset for visualization/validation (e.g., 20%)
    val_df = df_filtered.sample(frac=0.2, random_state=42)
    # Or take a fixed number for speed if just visualizing
    if len(val_df) > 500:
        val_df = val_df.sample(n=500, random_state=42)
    
    print(f"Visualizing on {len(val_df)} samples.")

    # Initialize Model
    model = BioCLIPClassifier(num_classes=len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    _, preprocess_val = model.get_transforms()
    
    val_dataset = AntDataset(val_df, preprocess_val, class_to_idx, args.base_path)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    # 2. Run Inference
    all_preds = []
    all_labels = []
    images_for_grid = []
    labels_for_grid = []
    preds_for_grid = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Running Inference"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
            # Save first batch for grid
            if len(images_for_grid) < 16:
                # Denormalize image for display
                # CLIP mean/std
                mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1).to(device)
                std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1).to(device)
                
                imgs_denorm = images * std + mean
                imgs_denorm = torch.clamp(imgs_denorm, 0, 1)
                
                for i in range(min(16 - len(images_for_grid), len(images))):
                    images_for_grid.append(imgs_denorm[i].cpu())
                    labels_for_grid.append(classes[labels[i]])
                    preds_for_grid.append(classes[predicted[i]])
                    
    # 3. Calculate Accuracy
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"Accuracy on validation set: {acc*100:.2f}%")
    
    # 4. Generate Grid Image (Like train_batch0.jpg)
    print("Generating validation grid...")
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i, ax in enumerate(axes.flat):
        if i < len(images_for_grid):
            img = images_for_grid[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            
            true_label = labels_for_grid[i]
            pred_label = preds_for_grid[i]
            
            color = 'green' if true_label == pred_label else 'red'
            title = f"T: {true_label}\nP: {pred_label}"
            
            # Truncate long names
            if len(title) > 50:
                title = title[:47] + "..."
                
            ax.set_title(title, color=color, fontsize=8)
        ax.axis('off')
        
    plt.tight_layout()
    grid_path = os.path.join(args.output_dir, "val_predictions.jpg")
    plt.savefig(grid_path)
    print(f"Saved predictions grid to {grid_path}")
    
    # 5. Generate Confusion Matrix
    # Only make it if classes < 50, otherwise it's too messy
    if len(classes) <= 60:
        print("Generating confusion matrix...")
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(20, 20))
        
        # Use simple matshow if seaborn missing, but try standard plot
        try:
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
            plt.savefig(cm_path)
            print(f"Saved confusion matrix to {cm_path}")
        except Exception as e:
            print(f"Could not generate heatmap: {e}")
            
    # Save Report
    report_path = os.path.join(args.output_dir, "accuracy_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.csv_file}\n")
        f.write(f"Validation Samples: {len(val_df)}\n")
        f.write(f"Accuracy: {acc*100:.2f}%\n")
    print(f"Saved report to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", default="combined_labels.csv")
    parser.add_argument("--base-path", default=".")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="results")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    visualize(args)
