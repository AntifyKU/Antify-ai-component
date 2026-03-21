FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6 \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
# Model (.pt) is downloaded from GCS at startup — not included in image (gitignored)
COPY . .

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/bioclip_finetuned.pt
# Set this in Cloud Run → Variables & Secrets:
# MODEL_GCS_PATH=gs://your-bucket/models/bioclip_finetuned.pt

EXPOSE 8080

# Run download_model.py first (downloads from GCS if MODEL_GCS_PATH is set),
# then start the server. This is synchronous — model is ready before uvicorn starts.
CMD python download_model.py && uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
