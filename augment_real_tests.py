import cv2
import os
import numpy as np
import random
from segmentation import segment_meter_digits

def augment_image(img):
    # img is already 28x28 or similar padded image, BGR
    # Let's resize it to 28x28 just in case
    img = cv2.resize(img, (28, 28))
    
    # Random slight rotation
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (28, 28), borderValue=(255, 255, 255))
    
    # Random slight translation
    tx = random.uniform(-2, 2)
    ty = random.uniform(-2, 2)
    M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(rotated, M_trans, (28, 28), borderValue=(255, 255, 255))
    
    # Random blur
    if random.random() < 0.5:
        translated = cv2.GaussianBlur(translated, (3, 3), 0)
        
    return translated

def extract_and_augment(img_path, true_digits):
    print(f"Extracting digits from {img_path} with true labels {true_digits}")
    img = cv2.imread(img_path)
    digits = segment_meter_digits(img, num_digits=len(true_digits))
    
    if len(digits) != len(true_digits):
        print(f"Failed to segment {len(true_digits)} digits. Found {len(digits)}")
        return
        
    for i, (digit_img, label) in enumerate(zip(digits, true_digits)):
        out_dir = os.path.join("datasets", "train", str(label))
        os.makedirs(out_dir, exist_ok=True)
        
        # Save original
        base_name = os.path.basename(img_path).replace(".jpg", "")
        cv2.imwrite(os.path.join(out_dir, f"{base_name}_orig_pos{i}.jpg"), digit_img)
        
        # Save augmentations
        for aug_i in range(100): # 100 augmented images per digit
            aug_img = augment_image(digit_img)
            cv2.imwrite(os.path.join(out_dir, f"{base_name}_aug_{i}_{aug_i}.jpg"), aug_img)
            
if __name__ == "__main__":
    os.makedirs("datasets/train", exist_ok=True)
    extract_and_augment("test_1.jpg", "00001")
    extract_and_augment("test_2.jpg", "00002")
    extract_and_augment("image.png", "00234")
    extract_and_augment("image copy.png", "97865")
    print("Real data extraction and augmentation complete!")
