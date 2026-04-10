import argparse
import inspect
import torch
import open_clip
from PIL import Image
import os
import sys

# Add current directory to path to find bioclip_model if run from scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bioclip_model import BioCLIPClassifier


def _safe_torch_load(path, map_location):
    """Load a checkpoint with secure deserialization only."""
    load_signature = inspect.signature(torch.load)
    if "weights_only" not in load_signature.parameters:
        raise RuntimeError(
            "This PyTorch version does not support safe checkpoint loading "
            "(requires torch.load(..., weights_only=True)). "
            "Please upgrade PyTorch to a version that supports weights_only."
        )

    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid checkpoint format in {path}: expected a dict.")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Invalid checkpoint in {path}: missing 'model_state_dict'.")
    return checkpoint

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
    # Skip header lines (detected by checking for 'Scientific Name' or similar)
    start_reading = False
    temp_classes = []
    
    # Check if this is the structured file with headers
    has_header = any("Scientific Name" in line for line in lines)
    
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
         # Fallback for simple list if header logic failed but we didn't explicitly detect it
         # This covers cases where we might have missed the condition above
         for line in lines:
            line = line.strip()
            if line and not line.startswith("#"):
                temp_classes.append(line)

    # Remove duplicates and sort
    classes = sorted(list(set(temp_classes)))
    return classes

def main():
    parser = argparse.ArgumentParser(description="BioCLIP Inference for Ant Species")
    parser.add_argument("--image", required=True, help="Path to the image file")
    # Default to the classes file generated during training
    parser.add_argument("--class-file", default="models/classes.txt", help="Path to file containing class names")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to show")
    parser.add_argument("--model-path", default=None, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.01, help="Confidence threshold for detection (0.0-1.0)")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.model_path:
        print(f"Loading fine-tuned model from {args.model_path}...")
        
        if not os.path.exists(args.model_path):
             print(f"Error: Model file {args.model_path} not found.")
             return

        # Load checkpoint first to see if it has classes
        checkpoint = _safe_torch_load(args.model_path, map_location=device)
        
        if 'classes' in checkpoint:
            classes = checkpoint['classes']
            print(f"Loaded {len(classes)} classes from model checkpoint.")
        else:
            # Fallback to classes.txt
            model_dir = os.path.dirname(args.model_path)
            class_file = os.path.join(model_dir, "classes.txt")
            if not os.path.exists(class_file):
                print(f"Error: classes.txt not found in {model_dir}. Required for fine-tuned model.")
                return
    
            with open(class_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
                
            print(f"Loaded {len(classes)} classes from {class_file}.")
        
        # Load Model
        model = BioCLIPClassifier(num_classes=len(classes))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        _, preprocess = model.get_transforms()
        image = preprocess(Image.open(args.image)).unsqueeze(0).to(device)
        
        # --- Safety Check (Is it even an ant?) ---
        # The fine-tuned head forces a choice among ants. We use the backbone to check if it's an ant first.
        tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
        
        # Expanded prompts to catch diverse ant images (specimens, colonies, macro, etc.)
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
            "cartoon", "drawing", "illustration", "clipart", "digital art", "vector graphics"
        ]
        
        safety_prompts = positive_prompts + negative_prompts
        safety_tokens = tokenizer(safety_prompts).to(device)
        
        with torch.no_grad():
            img_features = model.backbone.encode_image(image)
            img_features /= img_features.norm(dim=-1, keepdim=True)
            
            text_features = model.backbone.encode_text(safety_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Zero-shot safety probabilities
            safety_probs = (100.0 * img_features @ text_features.T).softmax(dim=-1)
            
            # Get top score for "Ant/Insect" group vs "Non-Ant" group
            positive_score = safety_probs[0][:len(positive_prompts)].sum().item()
            negative_score = safety_probs[0][len(positive_prompts):].sum().item()
            
            # Find specific top detection for logging
            top_idx = safety_probs[0].argmax().item()
            top_prob = safety_probs[0][top_idx].item()
            
            # Conservative Rejection:
            # Only reject if Negative Score is significantly higher than Positive Score
            # AND the top specific detection is a known non-ant object with high confidence
            
            if negative_score > 0.6 and top_idx >= len(positive_prompts):
                detected_obj = safety_prompts[top_idx]
                print(f"\n[Safety Gate] Rejected: Detected '{detected_obj}' ({top_prob*100:.2f}%) instead of an ant.")
                print(f"(Ant Score: {positive_score*100:.1f}% vs Non-Ant Score: {negative_score*100:.1f}%)")
                return 

            # --- Proceed to Fine-Tuned Classification ---
            logits = model(image)
            probs = torch.softmax(logits, dim=1)
            
        top_probs, top_labels = probs.cpu().topk(args.top_k, dim=1)
        
        # Check threshold
        top_prob = top_probs[0][0].item()
        if top_prob < args.threshold:
            print(f"\nResult: No ant detected (Confidence: {top_prob*100:.2f}% < {args.threshold*100:.0f}%).")
            print("Closest matches (unreliable):")
        else:
            print("\nTop predictions:")
            
        for i in range(args.top_k):
            # probs[0][i].item() is a float like 0.457 (45.7%), so we multiply by 100.
            prob_percent = top_probs[0][i].item() * 100
            print(f"{classes[top_labels[0][i].item()]}: {prob_percent:.2f}%")
            
        return # Exit main function after printing fine-tuned results
        
    else:
        # Zero-shot mode
        model, preprocess, tokenizer = load_bioclip_model()
        model = model.to(device)
        
        classes = load_classes(args.class_file)
        if not classes:
            print("No classes found. Using a default formatted list from top 50 thai ants for testing.")
            # Fallback list of top 50 Thai ants
            classes = [
                "Aenictus", "Anochetus", "Anoplolepis gracilipes", "Aphaenogaster", "Camponotus", 
                "Camponotus albosparsus", "Camponotus auriventris", "Camponotus camelinus", "Camponotus carin", 
                "Camponotus festinus", "Camponotus misturus", "Camponotus mutilarius", "Camponotus nicobarensis", 
                "Camponotus parius", "Camponotus rufoglaucus", "Camponotus singularis", "Carebara", 
                "Carebara affinis", "Carebara castanea", "Carebara diversa", "Cataulacus", 
                "Cataulacus granulatus", "Crematogaster", "Crematogaster aurita", "Crematogaster rogenhoferi", 
                "Diacamma", "Diacamma rugosum", "Dolichoderus", "Dolichoderus thoracicus", "Dorylus", 
                "Dorylus laevigatus", "Dorylus orientalis", "Ectomomyrmex", "Gnamptogenys", 
                "Iridomyrmex", "Iridomyrmex anceps", "Leptogenys", "Leptogenys diminuta", "Leptogenys kitteli", 
                "Meranoplus", "Meranoplus bicolor", "Monomorium", "Monomorium chinense", "Monomorium destructor", 
                "Monomorium floricola", "Monomorium pharaonis", "Odontomachus", "Odontomachus monticola", 
                "Odontomachus rixosus", "Odontomachus simillimus", "Oecophylla smaragdina", "Paratrechina longicornis", 
                "Pheidole", "Pheidole megacephala", "Pheidole parva", "Plagiolepis", "Polyrhachis", 
                "Polyrhachis abdominalis", "Polyrhachis armata", "Polyrhachis bicolor", "Polyrhachis dives", 
                "Polyrhachis furcata", "Polyrhachis illaudata", "Polyrhachis proxima", "Polyrhachis rastellata", 
                "Polyrhachis saevissima", "Pristomyrmex", "Solenopsis", "Solenopsis geminata", "Tapinoma", 
                "Tapinoma melanocephalum", "Technomyrmex", "Technomyrmex albipes", "Tetramorium", 
                "Tetramorium bicarinatum", "Tetramorium lanuginosum", "Tetramorium smithei", "Tetraponera", 
                "Tetraponera allaborans", "Tetraponera nigra", "Tetraponera rufonigra"
            ]
            
        # Add negative classes to reduce false positives
        negative_classes = [
            "not an ant", "random noise", "human", "cat", "dog", "car", "building", "food", "plant", "other object",
            "cartoon", "drawing", "illustration", "clipart", "digital art", "vector graphics",
            "cartoon of an ant", "drawing of an ant", "ant illustration", "specimen illustration"
        ]
        all_classes = classes + negative_classes
            
        print(f"Loaded {len(classes)} ant classes and {len(negative_classes)} negative classes.")
        
        image = preprocess(Image.open(args.image)).unsqueeze(0).to(device)
        text = tokenizer(all_classes).to(device)
        
        with torch.no_grad(), torch.amp.autocast(device_type=device.type):
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
        top_probs, top_labels = text_probs.cpu().topk(args.top_k, dim=-1)
    
    print("\nTop predictions:")
    for i in range(args.top_k):
        label_idx = top_labels[0][i].item()
        prob_percent = top_probs[0][i].item() * 100
        
        if not args.model_path and label_idx >= len(classes):
             class_name = all_classes[label_idx]
             print(f"[Non-Ant] {class_name}: {prob_percent:.2f}%")
             if i == 0:
                 print("\nResult: No ant detected.")
        else:
             class_name = classes[label_idx] if args.model_path else all_classes[label_idx]
             print(f"{class_name}: {prob_percent:.2f}%")

if __name__ == "__main__":
    main()
