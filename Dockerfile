FROM python:3.10-slim
WORKDIR /app

# Install system dependencies (including gcloud CLI for model download)
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  gnupg \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6 \
  && echo "deb [signed-by=/usr/share/keyrings/cloud.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
  | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
  && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | gpg --dearmor -o /usr/share/keyrings/cloud.gpg \
  && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (model .pt files are gitignored, downloaded at runtime)
COPY . .

ENV PYTHONUNBUFFERED=1

# MODEL_GCS_PATH must be set as an env var in Cloud Run, e.g.:
#   gs://your-bucket/models/bioclip_finetuned.pt
ENV MODEL_GCS_PATH=""
ENV MODEL_PATH="/app/models/bioclip_finetuned.pt"

EXPOSE 8080

# Download model from GCS at startup, then launch server
CMD mkdir -p /app/models && \
  echo "Downloading model from ${MODEL_GCS_PATH}..." && \
  gsutil cp "${MODEL_GCS_PATH}" "${MODEL_PATH}" && \
  echo "Model downloaded. Starting server on port ${PORT:-8080}..." && \
  uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
