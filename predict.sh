#!/bin/bash
echo "Starting BioCLIP Ant Predictor..."
python scripts/interactive_inference.py --model-path "models/bioclip_finetuned.pt"
