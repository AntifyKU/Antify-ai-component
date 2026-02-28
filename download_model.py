"""
Download the model from GCS before starting the server.
Reads MODEL_GCS_PATH env var, e.g.: gs://antify-models/models/bioclip_finetuned.pt
"""
import os
import sys
from pathlib import Path

def download_model():
    gcs_path = os.environ.get("MODEL_GCS_PATH", "").strip()
    model_path = os.environ.get("MODEL_PATH", "/app/models/bioclip_finetuned.pt")

    if not gcs_path:
        print("[download_model] MODEL_GCS_PATH not set — skipping download.")
        return

    if os.path.exists(model_path):
        print(f"[download_model] Model already exists at {model_path} — skipping download.")
        return

    # Parse gs://bucket/path/to/file
    if not gcs_path.startswith("gs://"):
        print(f"[download_model] Invalid MODEL_GCS_PATH: {gcs_path}", file=sys.stderr)
        sys.exit(1)

    gcs_path_stripped = gcs_path[len("gs://"):]
    bucket_name, _, blob_name = gcs_path_stripped.partition("/")

    print(f"[download_model] Downloading from bucket={bucket_name} blob={blob_name} -> {model_path}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(model_path)

    print(f"[download_model] Done! Model saved to {model_path}")


if __name__ == "__main__":
    download_model()
