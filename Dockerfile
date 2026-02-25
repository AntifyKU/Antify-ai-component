FROM python:3.10-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6 \
  && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY . .

ENV PYTHONUNBUFFERED=1

# Cloud Run sets $PORT at runtime (default 8080)
EXPOSE 8080

# Use shell form so ${PORT} env var is expanded at runtime
CMD uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}
