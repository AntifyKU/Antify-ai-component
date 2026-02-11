# Antify AI Component: BioCLIP Ant Classifier

🐜 **BioCLIP Ant Classifier** - A fine-tuned BioCLIP model for identifying ant species from images.

## Features

- **Species Classification**: Identifies 170+ ant species.
- **Safety Gate**: Rejects non-ant images (drawings, cartoons, other insects) to reduce false positives.
- **Interactive Mode**: Fast, continuous prediction without reloading the model.
- **Deployment Ready**: Self-contained inference scripts in `deployment/`.

## Quick Start

### Windows
Double-click **`predict.bat`** to start the interactive predictor.

### Linux/Mac
Run:
```bash
./predict.sh
```

### Manual Command
```bash
python scripts/interactive_inference.py --model-path "models/bioclip_finetuned.pt"
```

## Usage

1.  Start the script.
2.  Wait for the model to load.
3.  Paste the path to an image (e.g., `images/test_ant.jpg`).
4.  Get immediate results!

## Deployment

To run this on another machine:
1.  Copy the `deployment/` folder.
2.  Download/Move `models/bioclip_finetuned.pt` into that folder.
3.  Run `predict.bat` (or `predict.sh`).

## Model Info

- **Base Model**: [BioCLIP](https://huggingface.co/imageomics/bioclip) (ViT-B/16)
- **Training**: Fine-tuned on Thai Ant Dataset + Global AntWeb data.
- **Method**: Frozen backbone with Zero-Shot initialization (Safe Training).

## License

MIT License
