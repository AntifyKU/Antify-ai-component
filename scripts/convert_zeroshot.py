import torch
import os
import sys
import open_clip
from bioclip_model import BioCLIPClassifier
from bioclip_inference import load_classes, load_bioclip_model

def convert_zeroshot_to_model():
    print("Converting Zero-Shot to Static Model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Classes
    class_file = "top50_thai_ants.txt"
    classes = load_classes(class_file)
    
    # Add negative classes (matching inference script)
    negative_classes = ["not an ant", "random noise", "human", "cat", "dog", "car", "building", "food", "plant", "other object"]
    all_classes = classes + negative_classes
    
    print(f"Total classes: {len(all_classes)} ({len(classes)} ants + {len(negative_classes)} negatives)")
    
    # 2. Load BioCLIP (CLIP model)
    clip_model, _, tokenizer = load_bioclip_model()
    clip_model = clip_model.to(device)
    clip_model.eval()
    
    # 3. Compute Text Embeddings (The "Weights")
    print("Computing text embeddings...")
    text_tokens = tokenizer(all_classes).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 4. Create Classifier Model
    print("Creating BioCLIPClassifier...")
    # Initialize with same backbone
    model = BioCLIPClassifier(num_classes=len(all_classes)).to(device)
    
    # Set the backbone state (share weights)
    model.backbone.load_state_dict(clip_model.state_dict())
    
    # Set the Classifier Head weights to the Text Embeddings
    # Note: BioCLIP inference usage multiplies by 100.0 (temperature scaling)
    # So we bake this into the weights: Weight = TextFeature * 100.0
    model.classifier.weight.data = text_features.float() * 100.0
    model.classifier.bias.data.fill_(0.0)
    
    # 5. Save Model
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "bioclip_zeroshot.pt")
    
    print(f"Saving model to {save_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': all_classes
    }, save_path)
    
    # Save class list text file for inference script compatibility
    class_save_path = os.path.join(output_dir, "classes_zeroshot.txt")
    with open(class_save_path, 'w') as f:
        for cls in all_classes:
            f.write(f"{cls}\n")
            
    print(f"Saved class list to {class_save_path}")
    print("Conversion Complete!")

if __name__ == "__main__":
    convert_zeroshot_to_model()
