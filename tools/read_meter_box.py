import os
import sys
import argparse
import logging

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.getLogger("absl").setLevel(logging.ERROR)

try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
    pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
from segmentation import segment_meter_digits
from inference_lib import WaterMeterReader

def read_full_meter(image_path, num_digits=5, model_path='water_meter_modern.keras'):
    """
    Read water meter digits from image.
    """
    print(f"--- Processing image: {image_path} ---", flush=True)
    
    # 1. Segment digit boxes
    try:
        digits_images = segment_meter_digits(image_path, num_digits=num_digits, margin_ratio=0.22)
        print(f"[INFO] Segmented into {len(digits_images)} digit boxes.", flush=True)
    except Exception as e:
        print(f"[ERROR] Image segmentation failed: {e}", flush=True)
        return None
        
    # 2. Initialize inference model
    try:
        reader = WaterMeterReader(model_path=model_path)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}", flush=True)
        return None
        
    # 3. Predict sequence with mechanical wheel logic
    final_number_str = ""
    print("\nDigit predictions:", flush=True)
    
    try:
        sequence_results = reader.predict_sequence(digits_images)
        
        if sequence_results is None:
            print("\n[WARNING] Digits obscured or contaminated (NaN detected).", flush=True)
            print("Please clean the meter glass and recapture.", flush=True)
            return "WARNING_NAN"
            
        for i, res in enumerate(sequence_results):
            digit_val = res['digit']
            conf = res['confidence']
            print(f"  - Position {i+1}: Digit = {digit_val} (Confidence: {conf*100:.1f}%)", flush=True)
            final_number_str += str(digit_val)
            
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}", flush=True)
        return None

    print(f"\nFinal meter reading: {final_number_str}", flush=True)
    return final_number_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read water meter digit sequence from image.")
    parser.add_argument("image_path", type=str, help="Path to meter image")
    parser.add_argument("--digits", type=int, default=5, help="Number of digits (default: 5)")
    parser.add_argument("--model", type=str, default="water_meter_modern.keras", help="Path to model file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"[ERROR] File not found: {args.image_path}", flush=True)
        sys.exit(1)
        
    read_full_meter(args.image_path, num_digits=args.digits, model_path=args.model)
