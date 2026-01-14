"""
Download ant detection model from Roboflow.
Run this script to download a pre-trained YOLOv8 ant detection model.
"""

from roboflow import Roboflow
import os

# Your API key
API_KEY = "G9ptnLiK8yXUHfL8UM1b"

print("Connecting to Roboflow...")
rf = Roboflow(api_key=API_KEY)

print("Finding ant detection project...")
try:
    # Steam RE ant detection project
    project = rf.workspace("steam-re").project("ant-detection")
    print(f"Found project: {project.name}")
    
    # Get available versions
    print(f"\nAvailable versions:")
    for v in project.versions():
        print(f"  Version {v.version}: {v.id}")
    
    # Get the model for inference - try version 8 (latest)
    print("\nGetting model for inference (version 8)...")
    model = project.version(8).model
    
    if model is None:
        print("Version 8 has no hosted model, trying version 7...")
        model = project.version(7).model
    
    if model is None:
        print("No hosted model available. Downloading weights instead...")
        # Download the dataset with weights
        version = project.version(8)
        dataset = version.download("yolov8", location="models/ant_detection_roboflow")
        print(f"Downloaded to: models/ant_detection_roboflow/")
    else:
        print("\n✓ Model ready for inference!")
    print("\nTo detect ants in an image, use:")
    print('  model.predict("your_image.jpg", confidence=40).save("result.jpg")')
    
    # Test on a sample image
    test_image = "images/Oecophylla_smaragdina/333558933_specimen.jpg"
    if os.path.exists(test_image):
        print(f"\nTesting on: {test_image}")
        prediction = model.predict(test_image, confidence=40)
        prediction.save("ant_detection_result.jpg")
        print("✓ Result saved to: ant_detection_result.jpg")
        print(f"Predictions: {prediction.json()}")
    else:
        print(f"\nTest image not found: {test_image}")
        print("You can test with: model.predict('path/to/image.jpg', confidence=40).save('result.jpg')")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
