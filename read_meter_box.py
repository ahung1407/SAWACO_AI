import os
# Tắt bớt log rác của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger("absl").setLevel(logging.ERROR)

import argparse
import sys
import cv2
from segmentation import segment_meter_digits
from inference_lib import WaterMeterReader

def read_full_meter(image_path, num_digits=5, model_path='water_meter_modern.keras'):
    """
    Doc toan bo day so tu anh khung dong ho nuoc.
    """
    print(f"--- Dang xu ly anh: {image_path} ---")
    
    # 1. Cắt ảnh thành các chữ số
    try:
        # Cắt bớt 22% viền để loại bỏ khung nhựa giữa các con số
        digits_images = segment_meter_digits(image_path, num_digits=num_digits, margin_ratio=0.22)
        print(f"[OK] Da cat thanh {len(digits_images)} chu so.")
    except Exception as e:
        print(f"[LOI] Phan tach anh that bai: {e}")
        return None
        
    # 2. Khởi tạo mô hình nhận diện
    try:
        reader = WaterMeterReader(model_path=model_path)
    except Exception as e:
        print(f"[LOI] Khong the load mo hinh: {e}")
        return None
        
    # 3. Dự đoán từng chữ số
    final_number_str = ""
    print("\nChi tiet du doan tung o:")
    
    for i, digit_img in enumerate(digits_images):
        # Lưu ra để debug nếu cần
        # cv2.imwrite(f"debug_digit_{i}.jpg", digit_img)
        
        try:
            # reader.predict nhận numpy array
            result = reader.predict(digit_img)
            digit_val = result['digit']
            conf = result['confidence']
            
            print(f"  - O so {i+1}: Nhan dien = {digit_val} (Do tin cay: {conf*100:.1f}%)")
            final_number_str += str(digit_val)
            
        except Exception as e:
            print(f"  - O so {i+1}: LOI du doan ({e})")
            final_number_str += "?"

    print(f"\n>>> KET QUA CUOI CUNG: {final_number_str}")
    return final_number_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Đọc khung dãy số đồng hồ nước.")
    parser.add_argument("image_path", type=str, help="Đường dẫn đến ảnh khung dãy số")
    parser.add_argument("--digits", type=int, default=5, help="Số lượng chữ số (mặc định: 5)")
    parser.add_argument("--model", type=str, default="water_meter_modern.keras", help="Đường dẫn đến model")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Lỗi: Không tìm thấy file ảnh {args.image_path}")
        sys.exit(1)
        
    read_full_meter(args.image_path, num_digits=args.digits, model_path=args.model)
