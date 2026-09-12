import os, sys, glob
import cv2
import numpy as np

# Force UTF-8 for console output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from segmentation import segment_meter_digits
from inference_lib import WaterMeterReader

def main():
    real_dir = "real_data_base"
    images = sorted(glob.glob(os.path.join(real_dir, "*.jpg")))
    
    print(f"Found {len(images)} images in {real_dir}")
    if not images:
        print("No images found!")
        return
        
    reader = WaterMeterReader('water_meter_modern.keras')
    
    debug_out_dir = "debug_real_test"
    os.makedirs(debug_out_dir, exist_ok=True)
    
    results = []
    
    print("\n" + "="*80)
    print(f"{'#':<3} | {'File Name':<35} | {'Reading':<8} | {'Confidences':<30}")
    print("="*80)
    
    for idx, img_path in enumerate(images, 1):
        fname = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"{idx:<3} | {fname:<35} | ERROR: Cannot read image")
            continue
            
        digits_imgs = segment_meter_digits(img)
        
        # Save segmented digit crops for visual inspection
        sub_dir = os.path.join(debug_out_dir, os.path.splitext(fname)[0])
        os.makedirs(sub_dir, exist_ok=True)
        for d_i, d_img in enumerate(digits_imgs):
            cv2.imwrite(os.path.join(sub_dir, f"digit_{d_i}.jpg"), d_img)
            
        preds = reader.predict_sequence(digits_imgs)
        if preds is None:
            reading_str = "NaN (Blocked)"
            conf_str = "N/A"
        else:
            reading_str = "".join(str(p['digit']) for p in preds)
            conf_str = " ".join(f"{p['confidence']*100:.0f}%" for p in preds)
            
        print(f"{idx:<3} | {fname:<35} | {reading_str:<8} | {conf_str:<30}")
        results.append({
            "idx": idx,
            "filename": fname,
            "reading": reading_str,
            "confidences": [p['confidence'] for p in preds] if preds else []
        })
        
    print("="*80)
    print(f"Testing completed for {len(results)} images.")

if __name__ == "__main__":
    main()
