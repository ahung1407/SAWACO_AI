
import cv2
import numpy as np
import os
import subprocess
import sys
import time

def create_water_meter_digit(digit, filename):
    """
    Creates a synthetic 'water meter' style digit image.
    Style: White digit on black background, simulating a mechanical roller.
    """
    # 1. Create canvas (Black background)
    h, w = 100, 80
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 2. Add a 'Roller' effect (slightly lighter gray borders top/bottom)
    # This simulates the curvature of the mechanical wheel
    img[0:10, :] = (30, 30, 30)
    img[h-10:h, :] = (30, 30, 30)
    
    # 3. Draw Digit
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 2.5
    color = (230, 230, 230) # Off-white
    thickness = 4
    
    # Center the text
    text_size = cv2.getTextSize(str(digit), font, font_scale, thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (h + text_size[1]) // 2
    
    cv2.putText(img, str(digit), (text_x, text_y), font, font_scale, color, thickness)
    
    # 4. Add some noise/blur to simulate low-res camera
    noise = np.random.normal(0, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Save
    cv2.imwrite(filename, img)
    return filename

def run_pipeline_test():
    print("=== AUTOMATED TEST SUITE: Water Meter Digits 0-9 ===\n")
    
    # 1. Generate Images
    print("[1/3] Generating Synthetic Data (0-9 Water Meter Style)...")
    test_files = []
    for i in range(10):
        fname = f"wm_digit_{i}.jpg"
        create_water_meter_digit(i, fname)
        test_files.append((i, fname))
    print(f"      Generated {len(test_files)} images.\n")

    # 2. Ensure Model is Trained (Auto-Train)
    print("[2/3] Auto-Training Model (Running pipeline without args)...")
    # We run the main pipeline script. It detects if model exists, but we want to ensure fresh state or just use it.
    # The user asked to "Auto Train", so let's run the training mode.
    # To force training, we can either delete the .h5 or just run it. 
    # My pipeline executes training if no args provided.
    
    try:
        subprocess.run([sys.executable, "water_meter_pipeline.py"], check=True)
        print("      Training/Setup Complete.\n")
    except subprocess.CalledProcessError as e:
        print(f"      CRITICAL ERROR during training: {e}")
        return

    # 3. Run Inference Test (Auto-Test)
    print("[3/3] Auto-Testing 0-9...")
    
    results = []
    correct_count = 0
    
    for digit, fname in test_files:
        print(f"  > Testing Digit {digit} (File: {fname})...", end=" ")
        
        # Run pipeline on image file
        # Capture stdout
        result = subprocess.run(
            [sys.executable, "water_meter_pipeline.py", fname], 
            capture_output=True, 
            text=True
        )
        
        output = result.stdout
        
        # Parse output for "PREDICTION RESULT: X"
        predicted = -1
        for line in output.splitlines():
            if ">>> PREDICTION RESULT:" in line:
                try:
                    predicted = int(line.split(":")[1].strip())
                except:
                    pass
        
        status = "FAIL"
        if predicted == digit:
            status = "PASS"
            correct_count += 1
        
        print(f"[{status}] -> Predicted: {predicted}")
        results.append((digit, predicted, status))
        
    # Summary
    print("\n=== TEST RESULTS SUMMARY ===")
    print(f"Accuracy: {correct_count}/10 ({correct_count*10}%)")
    for digit, pred, status in results:
        print(f"Digit {digit}: {status} (Pred: {pred})")
        
    if correct_count < 5:
        print("\nNote: Accuracy might be low because MNIST (handwritten) is very different from Mechanical digits.")
        print("To improve, we would eventually need a real dataset of mechanical meter digits or use Transfer Learning.")

if __name__ == "__main__":
    run_pipeline_test()
