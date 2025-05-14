# Ant Species Identification - Model Training
This project trains and tracks a machine learning model to identify ant species from images. It uses image data from iNaturalist and supports model versioning and experiment tracking using MLflow.

**📌 Currently working on the** `model-training` **branch**

## Project Structure

```
Antify-ai-component/
├── data/                         # Image dataset from iNaturalist
├── mlruns/                       # MLflow tracking files
├── ant_species_identification.ipynb  # Jupyter Notebook for model training
├── extract.py                    # Script to scrape or process images
├── requirements.txt             # Python dependencies
├── thai_species.csv             # CSV file with species metadata
├── .gitignore
├── LICENSE
└── README.md
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/AntifyKU/Antify-ai-component.git
cd Antify-ai-component
```

### 2. Switch to `model-training` Branch

```bash
git checkout model-training
```

### 3. Create and Activate a Virtual Environment

```bash
python -m venv env
source env/bin/activate   # On macOS and Linux   
env\Scripts\activate      # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Run the Image Extraction Script (Optional)

If you want to collect or reprocess image data:

```bash
python extract.py
```

Please don't execute this code again for the time being, as we already have some data.

### 2. Train the Model

You can train the model and track experiments using the notebook:

```bash
jupyter notebook ant_species_identification.ipynb
```

### 3. Launch MLflow UI (Optional)

To view experiment results:

```bash
mlflow ui
```

Open your browser and go to: [http://localhost:5000](http://localhost:5000)

## Data

* The `data/` folder contains images collected from iNaturalist.
* `thai_species.csv` includes metadata for Thai ant species.

## Dependencies

Main libraries used:

* `tensorflow`
* `pandas`
* `numpy`
* `scikit-learn`
* `mlflow`

See `requirements.txt` for the full list.
