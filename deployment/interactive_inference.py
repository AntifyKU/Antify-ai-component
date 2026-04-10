import argparse
import torch
import open_clip
from PIL import Image
import os
import sys

# Add current directory to path to find bioclip_model if run from scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bioclip_model import BioCLIPClassifier

def load_bioclip_model():
    print("Loading BioCLIP model...")
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
    return model, preprocess_val, tokenizer

def load_classes(class_file):
    classes = []
    if not os.path.exists(class_file):
        print(f"Warning: Class file {class_file} not found.")
        return classes
    
    try:
        with open(class_file, 'r', encoding='utf-16') as f:
            lines = f.readlines()
    except UnicodeError:
        try:
            with open(class_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except UnicodeError:
            print(f"Error: Could not decode {class_file} with utf-16 or utf-8.")
            return classes
    
    # Strategy 1: strict header parsing (legacy)
    temp_classes = []
    has_header = any("Scientific Name" in line for line in lines)
    start_reading = False
    
    if has_header:
        for i, line in enumerate(lines):
            if "Scientific Name" in line:
                start_reading = True
                continue
            
            if start_reading:
                if line.strip().startswith("---"):
                    if not temp_classes:
                        continue
                    break
                
                parts = line.split()
                if len(parts) >= 3 and parts[0].isdigit():
                    scientific_name = f"{parts[1]} {parts[2]}"
                    temp_classes.append(scientific_name)
    else:
        # Strategy 2: Simple list (one per line)
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                temp_classes.append(line)
                
    if not temp_classes and not has_header:
         for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                temp_classes.append(line)

    classes = sorted(list(set(temp_classes)))
    return classes

def run_inference(model, preprocess, tokenizer, classes, image_path, device, args, all_classes=None):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return

    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # --- Safety Check ---
    positive_prompts = [
        "ant", "insect", "bug", "ant colony", "ants on a leaf", 
        "macro photo of an ant", "specimen of an ant", "Hymenoptera",
        "winged ant", "ant queen", "hairy ant", "black ant",
        "ant specimen", "ant on white background", "pinned ant", 
        "microscope photo of an ant", "yellow ant", "orange ant", 
        "ant on a leaf", "nature photo of an ant", "wild ant"
    ]
    negative_prompts = [
        "cat", "dog", "person", "human face",
        "cartoon", "drawing", "illustration", "clipart", "digital art", "vector graphics",
        "cartoon of an ant", "drawing of an ant", "ant illustration", "specimen illustration"
    ]
    
    safety_prompts = positive_prompts + negative_prompts
    safety_tokens = tokenizer(safety_prompts).to(device)
    
    with torch.no_grad():
        if hasattr(model, 'backbone'):
            # Fine-tuned model (BioCLIPClassifier wrapper)
            img_features = model.backbone.encode_image(image)
            text_features = model.backbone.encode_text(safety_tokens)
        else:
            # Zero-shot model (Raw CLIP)
            with torch.amp.autocast(device_type=device.type):
                img_features = model.encode_image(image)
                text_features = model.encode_text(safety_tokens)

        img_features /= img_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Zero-shot safety probabilities

        safety_probs = (100.0 * img_features @ text_features.T).softmax(dim=-1)
        
        # Get top score for "Ant/Insect" group vs "Non-Ant" group
        negative_score = safety_probs[0][len(positive_prompts):].sum().item()
        
        # Find specific top detection for logging
        top_idx = safety_probs[0].argmax().item()
        top_prob = safety_probs[0][top_idx].item()
        
        # Conservative Rejection
        if negative_score > 0.6 and top_idx >= len(positive_prompts):
            detected_obj = safety_prompts[top_idx]
            print(f"\n[Safety Gate] Rejected: Detected '{detected_obj}' ({top_prob*100:.2f}%) instead of an ant.")
            return 

        # --- Proceed to Fine-Tuned Classification ---
        if args.model_path:
             logits = model(image)
             probs = torch.softmax(logits, dim=1)
             top_probs, top_labels = probs.cpu().topk(args.top_k, dim=1)
        else:
             # Zero-shot
             text = tokenizer(all_classes).to(device)
             with torch.amp.autocast(device_type=device.type):
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_probs = (100.0 * img_features @ text_features.T).softmax(dim=-1)
             top_probs, top_labels = text_probs.cpu().topk(args.top_k, dim=-1)

    print("\nTop predictions:")
    for i in range(args.top_k):
        label_idx = top_labels[0][i].item()
        prob_percent = top_probs[0][i].item() * 100
        
        if not args.model_path and label_idx >= len(classes):
                class_name = all_classes[label_idx]
                print(f"[Non-Ant] {class_name}: {prob_percent:.2f}%")
        else:
                class_name = classes[label_idx] if args.model_path else all_classes[label_idx]
                print(f"{class_name}: {prob_percent:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="BioCLIP Interactive Inference")
    parser.add_argument("--class-file", default="models/classes.txt", help="Path to file containing class names")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to show")
    parser.add_argument("--model-path", default=None, help="Path to fine-tuned model checkpoint")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Load Model Once ---
    if args.model_path:
        print(f"Loading fine-tuned model from {args.model_path}...")
        if not os.path.exists(args.model_path):
             print(f"Error: Model file {args.model_path} not found.")
             return

        checkpoint = torch.load(args.model_path, map_location=device)
        if 'classes' in checkpoint:
            classes = checkpoint['classes']
            print(f"Loaded {len(classes)} classes from model checkpoint.")
        else:
             # Fallback
             classes = load_classes(args.class_file) # Simplified fallback for interactive
             print(f"Loaded {len(classes)} classes from file.")

        model = BioCLIPClassifier(num_classes=len(classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        _, preprocess = model.get_transforms()
        tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
        all_classes = None
        
    else:
        # Zero-shot mode initialization
        model, preprocess, tokenizer = load_bioclip_model()
        model = model.to(device)
        model.eval()
        classes = load_classes(args.class_file)
        if not classes:
             # Hardcoded fallback list (abbreviated)
             classes = ["Oecophylla smaragdina", "Solenopsis geminata", "Carebara diversa"] 
             print("Warning: classes.txt not found, using minimal fallback.")
             
        negative_classes = [
            "not an ant", "random noise", "human", "cat", "dog", "car", "building", "food", "plant", "other object",
            "cartoon", "drawing", "illustration", "clipart", "digital art", "vector graphics",
            "cartoon of an ant", "drawing of an ant", "ant illustration", "specimen illustration"
        ]
        all_classes = classes + negative_classes
        print(f"Loaded {len(classes)} ant classes.")

    print("\n" + "="*50)
    print(" INTERACTIVE INFERENCE MODE")
    print(" Type an image path and press Enter.")
    print(" Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        try:
            image_path = input("Image Path > ").strip()
            if image_path.lower() in ["exit", "quit"]:
                break
            if not image_path:
                continue
            
            # Remove quotes if user dragged & dropped file
            image_path = image_path.strip('"').strip("'")
            
            run_inference(model, preprocess, tokenizer, classes, image_path, device, args, all_classes)
            print("-" * 30)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
