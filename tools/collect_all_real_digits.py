import os, sys, glob
import cv2

sys.path.append('.')
from segmentation import segment_meter_digits

GROUND_TRUTHS = {
    "device_xiao_s3_20260909_094142.jpg": [0, 0, 0, 0, 1],
    "device_xiao_s3_20260909_095354.jpg": [0, 0, 0, 0, 2],
    "device_xiao_s3_20260909_100525.jpg": [0, 0, 0, 0, 3],
    "device_xiao_s3_20260909_101854.jpg": [0, 0, 0, 0, 4],
    "device_xiao_s3_20260909_102216.jpg": [0, 0, 0, 0, 4],
    "device_xiao_s3_20260909_102432.jpg": [0, 0, 0, 0, 5],
    "device_xiao_s3_20260909_102638.jpg": [0, 0, 0, 0, 6],
    "device_xiao_s3_20260909_102856.jpg": [0, 0, 0, 0, 7],
    "device_xiao_s3_20260909_103034.jpg": [0, 0, 0, 0, 8],
    "device_xiao_s3_20260909_103222.jpg": [0, 0, 0, 0, 9],
    "device_xiao_s3_20260909_103920.jpg": [0, 0, 0, 1, 0],
    "device_xiao_s3_20260909_104204.jpg": [0, 0, 0, 1, 1],
    "device_xiao_s3_20260909_104558.jpg": [0, 0, 0, 1, 2],
    "device_xiao_s3_20260909_104858.jpg": [0, 0, 0, 2, 1],
    "device_xiao_s3_20260909_105206.jpg": [0, 0, 0, 3, 2],
    "device_xiao_s3_20260909_105530.jpg": [0, 0, 0, 4, 6],
    "device_xiao_s3_20260909_105724.jpg": [0, 0, 0, 5, 1],
    "device_xiao_s3_20260909_105930.jpg": [0, 0, 0, 6, 3],
    "device_xiao_s3_20260909_110142.jpg": [0, 0, 0, 7, 4],
    "device_xiao_s3_20260909_110538.jpg": [0, 0, 0, 8, 9],
    "device_xiao_s3_20260909_111235.jpg": [0, 0, 0, 9, 1],
    "device_xiao_s3_20260909_111515.jpg": [0, 0, 1, 2, 4],
    "device_xiao_s3_20260909_111955.jpg": [0, 0, 2, 8, 1],
}

def main():
    base_dir = "real_data_base"
    out_dir = "data/real_digits_labeled"
    
    # Ensure folders for all 10 digits
    for digit in range(10):
        os.makedirs(os.path.join(out_dir, str(digit)), exist_ok=True)
        
    count_per_digit = {d: 0 for d in range(10)}
    
    for fname, digits in GROUND_TRUTHS.items():
        img_path = os.path.join(base_dir, fname)
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        crops = segment_meter_digits(img)
        
        base_name = os.path.splitext(fname)[0]
        for idx, (label, crop) in enumerate(zip(digits, crops)):
            save_path = os.path.join(out_dir, str(label), f"{base_name}_d{idx}.jpg")
            cv2.imwrite(save_path, crop)
            count_per_digit[label] += 1
            
    print("="*60)
    print("HOAN THANH TRICH XUAT TOAN BO CHU SO THUC TE:")
    for d in range(10):
        print(f" - Chu so {d}: {count_per_digit[d]} mau thuc te")
    print("="*60)

if __name__ == "__main__":
    main()
