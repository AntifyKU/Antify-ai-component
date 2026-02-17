import torch
import os

MODEL_PATH = "models/bioclip_finetuned.pt"

def load_and_print_model_info():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("--- BioCLIP Model Info ---")
    try:
        if not torch.cuda.is_available():
            map_location = torch.device('cpu')
        else:
            map_location = torch.device('cuda')

        checkpoint = torch.load(MODEL_PATH, map_location=map_location)
        
        # Accuracy
        if "best_acc" in checkpoint:
            print(f"Best Validation Accuracy: {checkpoint['best_acc']:.2f}%")
        else:
            print("Best Validation Accuracy: Not found in checkpoint")

        # Epoch
        if "epoch" in checkpoint:
            print(f"Epoch Saved: {checkpoint['epoch'] + 1}") 
        
        # Classes
        if "classes" in checkpoint:
            classes = checkpoint["classes"]
            print(f"Number of Classes: {len(classes)}")
            print(f"Example Classes: {', '.join(classes[:5])}...")
        elif "class_to_idx" in checkpoint:
             print(f"Number of Classes: {len(checkpoint['class_to_idx'])}")

        print("---------------------------")

    except Exception as e:
        print(f"Failed to load checkpoint: {e}")

if __name__ == "__main__":
    load_and_print_model_info()
