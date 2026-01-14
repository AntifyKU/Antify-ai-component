"""
Test script for Ant Detection API.
Run this to test the API with a sample image.

Usage:
    python test_api.py                          # Uses default test image
    python test_api.py path/to/image.jpg        # Uses specified image
"""

import sys
import requests
from pathlib import Path


API_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("🔍 Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    return response.status_code == 200


def test_detect(image_path: str):
    """Test detection endpoint with an image."""
    print(f"\n🐜 Testing /detect endpoint with: {image_path}")
    
    if not Path(image_path).exists():
        print(f"   ❌ File not found: {image_path}")
        return False
    
    with open(image_path, "rb") as f:
        files = {"file": (Path(image_path).name, f, "image/jpeg")}
        response = requests.post(f"{API_URL}/detect", files=files)
    
    print(f"   Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"   ✅ Success!")
        print(f"   Message: {result['message']}")
        print(f"   Detections found: {result['num_detections']}")
        
        for det in result['detections']:
            print(f"      - {det['class_name']}: {det['confidence']:.2%} at {det['bbox']}")
        
        return True
    else:
        print(f"   ❌ Error: {response.text}")
        return False


def find_sample_image():
    """Find a sample image to test with."""
    # Look for any image in the images directory
    images_dir = Path("images")
    if images_dir.exists():
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            images = list(images_dir.rglob(ext))
            if images:
                return str(images[0])
    return None


def main():
    print("=" * 60)
    print("🐜 Ant Detection API Test")
    print("=" * 60)
    
    # Test health
    if not test_health():
        print("\n❌ API is not running! Start it with:")
        print("   uvicorn api.app:app --host 0.0.0.0 --port 8000")
        return
    
    # Get image path
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = find_sample_image()
        if not image_path:
            print("\n⚠️  No sample image found. Please provide one:")
            print("   python test_api.py path/to/image.jpg")
            return
    
    # Test detection
    test_detect(image_path)
    
    print("\n" + "=" * 60)
    print("📚 API Documentation: http://localhost:8000/docs")
    print("=" * 60)


if __name__ == "__main__":
    main()
