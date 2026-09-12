
from inference_lib import WaterMeterReader
import cv2
import glob

# 1. Initialize the Reader (Loads model once)
try:
    reader = WaterMeterReader(model_path='water_meter_lenet5.h5')
except Exception as e:
    print(f"Initialization Error: {e}")
    exit()

# 2. Simulate processing a stream of images
print("\n--- Processing Images ---")

# Example: Get all jpg files in datasets/test/3/ (just as a demo source)
demo_images = glob.glob('datasets/test/3/*.jpg')[:5]

if not demo_images:
    print("No images found to test. Running on synthetic files if available...")
    demo_images = ['test_digit_3.jpg', 'wm_digit_3.jpg']

for img_path in demo_images:
    try:
        # Perform Prediction
        result = reader.predict(img_path)
        
        print(f"Image: {img_path}")
        print(f"  -> Predicted Digit: {result['digit']}")
        print(f"  -> Confidence: {result['confidence']:.2f}")
        
    except Exception as e:
        print(f"Skipping {img_path}: {e}")

print("\n--- Real-time Integration Hint ---")
print("You can also pass numpy arrays directly (e.g., from a camera feed):")
print("frame = cv2.imread(...)")
print("result = reader.predict(frame)")
