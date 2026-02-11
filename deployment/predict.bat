@echo off
echo Starting BioCLIP Ant Predictor...
python interactive_inference.py --model-path "bioclip_finetuned.pt"
if %errorlevel% neq 0 (
    echo.
    echo Error running script. Please make sure you are in the correct environment.
    pause
)
