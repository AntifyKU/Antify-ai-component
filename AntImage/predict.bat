@echo off
echo Starting BioCLIP Ant Predictor...
python scripts/interactive_inference.py --model-path "models/bioclip_finetuned.pt"
if %errorlevel% neq 0 (
    echo.
    echo Error running script. Please make sure you are in the correct environment.
    pause
)
