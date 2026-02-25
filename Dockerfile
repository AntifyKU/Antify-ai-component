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
# (models/*.pt are gitignored — downloaded from GCS at startup via MODEL_GCS_PATH)
COPY . .

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/bioclip_finetuned.pt

EXPOSE 8080

# 1. Download model from GCS (if MODEL_GCS_PATH is set)
# 2. Start the server on the port Cloud Run provides via $PORT
CMD python download_model.py && uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
