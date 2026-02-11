# BioCLIP Ant Inference

This package allows you to run BioCLIP inference to classify ant species from images. It supports both a fine-tuned model and zero-shot classification.

## Setup

1.  **Install Python 3.8+**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to install PyTorch separately depending on your CUDA version. See [pytorch.org](https://pytorch.org))*

3.  **Get the Model:**
    - Place your fine-tuned model file (`bioclip_finetuned.pt`) in this folder.
    - Or use the zero-shot mode (requires internet to download BioCLIP weights initially).

## Usage

### 1. Interactive Mode (Fastest)

**Double-click `predict.bat`** (Windows) or run `./predict.sh` (Linux/Mac) to start immediately!

Or run via command line:
```bash
python interactive_inference.py --model-path bioclip_finetuned.pt
```
*If you don't have a fine-tuned model, omit `--model-path` to run in Zero-Shot mode.*

### 2. Single Image Mode

```bash
python bioclip_inference.py --image "path/to/image.jpg" --model-path bioclip_finetuned.pt
```

## Files
- `interactive_inference.py`: recommended script for checking multiple images.
- `bioclip_inference.py`: script for single-image checks or automation.
- `bioclip_model.py`: model definition (do not run directly).
- `classes_example.txt`: example of the class list format used.
