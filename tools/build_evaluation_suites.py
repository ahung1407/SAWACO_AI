"""
tools/build_evaluation_suites.py
---------------------------------
Xây dựng 3 Tập Kiểm thử Độc lập theo Chuẩn Nghiệp vụ Ngành Nước (Sawaco):

1. Suite 1: Real Baseline (23 ảnh chụp thực tế từ camera ESP32-S3 ngoài hiện trường)
2. Suite 2: Billing Digits Balanced (100 ảnh phủ đều 0000 - 9999 m3 trên 4 ô đen tính tiền nước)
3. Suite 3: Billing Jump Transitions & Floor Rule (50 ảnh kiểm tra bước nhảy 9 -> 0 và quy tắc làm tròn sàn bảo vệ khách hàng)
"""

import os
import sys
import glob
import json
import random
import shutil
import cv2
import numpy as np

# Đảm bảo in tiếng Việt trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tools.generate_synthetic_from_real import load_clean_masks, get_meter_window_boxes, render_stroke

ROOT_TEST_SUITES = "data/test_suites"
REAL_DATA_DIR = "real_data_base"
REAL_GROUND_TRUTH = {
    "device_xiao_s3_20260909_094142.jpg": "00001",
    "device_xiao_s3_20260909_095354.jpg": "00002",
    "device_xiao_s3_20260909_100525.jpg": "00003",
    "device_xiao_s3_20260909_101854.jpg": "00004",
    "device_xiao_s3_20260909_102216.jpg": "00004",
    "device_xiao_s3_20260909_102432.jpg": "00005",
    "device_xiao_s3_20260909_102638.jpg": "00006",
    "device_xiao_s3_20260909_102856.jpg": "00007",
    "device_xiao_s3_20260909_103034.jpg": "00008",
    "device_xiao_s3_20260909_103222.jpg": "00009",
    "device_xiao_s3_20260909_103920.jpg": "00010",
    "device_xiao_s3_20260909_104204.jpg": "00011",
    "device_xiao_s3_20260909_104558.jpg": "00012",
    "device_xiao_s3_20260909_104858.jpg": "00021",
    "device_xiao_s3_20260909_105206.jpg": "00032",
    "device_xiao_s3_20260909_105530.jpg": "00046",
    "device_xiao_s3_20260909_105724.jpg": "00051",
    "device_xiao_s3_20260909_105930.jpg": "00063",
    "device_xiao_s3_20260909_110142.jpg": "00074",
    "device_xiao_s3_20260909_110538.jpg": "00089",
    "device_xiao_s3_20260909_111235.jpg": "00091",
    "device_xiao_s3_20260909_111515.jpg": "00124",
    "device_xiao_s3_20260909_111955.jpg": "00281",
}

def generate_multiroll_meter(reading_str, masks, base_img_path, roll_dict=None):
    """
    Sinh ảnh mặt số đồng hồ nước chân thực dựa trên ảnh canvas thực tế.
    roll_dict: dict {wheel_idx: roll_ratio} với wheel_idx từ 0 đến 4.
    """
    base_img = cv2.imread(base_img_path)
    if base_img is None:
        raise ValueError(f"Không thể mở ảnh nền: {base_img_path}")
        
    base_name = os.path.basename(base_img_path)
    calib_file = "data/real_window_boxes.json"
    if os.path.exists(calib_file):
        with open(calib_file, "r", encoding="utf-8") as f:
            calib = json.load(f)
            windows = calib.get(base_name, get_meter_window_boxes(base_img))
    else:
        windows = get_meter_window_boxes(base_img)
        
    result = base_img.copy()
    if roll_dict is None:
        roll_dict = {}
        
    for i, ((bx, by, bw, bh), char) in enumerate(zip(windows, reading_str)):
        digit_val = int(char)
        is_red = (i == 4)
        
        roi = result[by:by+bh, bx:bx+bw].copy()
        rh, rw = roi.shape[:2]
        
        # 1. Inpaint xóa sạch nét số cũ
        mask = np.zeros((rh, rw), dtype=np.uint8)
        pad_x, pad_y = 2, 2
        if not is_red:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            ink_zone = gray_roi[pad_y:rh-pad_y, pad_x:rw-pad_x]
            _, bin_ink = cv2.threshold(ink_zone, 150, 255, cv2.THRESH_BINARY_INV)
            mask[pad_y:rh-pad_y, pad_x:rw-pad_x] = bin_ink
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        else:
            central = roi[pad_y:rh-pad_y, pad_x:rw-pad_x]
            b, g, r = cv2.split(central)
            is_red_ink = (r.astype(int) - np.maximum(b, g).astype(int) > 10) | (np.min(central, axis=2) < 150)
            mask[pad_y:rh-pad_y, pad_x:rw-pad_x] = is_red_ink.astype(np.uint8) * 255
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
            
        inpainted = cv2.inpaint(roi, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        
        # 2. Render nét chữ số mới
        target_h = int(rh * 0.70)
        m_img = masks.get(digit_val)
        if m_img is not None:
            mh, mw = m_img.shape[:2]
            target_w = min(int(rw * 0.72), max(int(rw * 0.25), int(mw * (target_h / float(mh)))))
        else:
            target_w = int(rw * 0.65)
            
        roll_ratio = roll_dict.get(i, 0.0)
        next_val = (digit_val + 1) % 10 if roll_ratio > 0.0 else None
        
        stroke_bgra = render_stroke(digit_val, target_w, target_h, masks, 
                                    next_val=next_val, roll_ratio=roll_ratio)
        
        dx1 = (rw - target_w) // 2
        dy1 = (rh - target_h) // 2
        dx2 = dx1 + target_w
        dy2 = dy1 + target_h
        
        alpha = stroke_bgra[:, :, 3].astype(np.float32) / 255.0
        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        
        ink_color = np.array([22, 28, 208] if is_red else [28, 28, 32], dtype=np.float32)
        wheel_roi = inpainted[dy1:dy2, dx1:dx2].astype(np.float32)
        
        blended = wheel_roi * (1.0 - alpha) + ink_color * alpha
        inpainted[dy1:dy2, dx1:dx2] = np.clip(blended, 0, 255).astype(np.uint8)
        
        result[by:by+bh, bx:bx+bw] = inpainted
            
    return result

def build_suite_1():
    """Suite 1: Real Baseline (23 ảnh chụp thực tế ESP32-S3)"""
    out_dir = os.path.join(ROOT_TEST_SUITES, "suite_1_real_baseline")
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    
    real_files = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "*.jpg")))
    for path in real_files:
        fname = os.path.basename(path)
        dest_path = os.path.join(out_dir, fname)
        shutil.copy2(path, dest_path)
        gt_full = REAL_GROUND_TRUTH.get(fname, "00000")
        gt_billing = gt_full[:4]
        manifest.append({
            "file": fname,
            "ground_truth_full": gt_full,
            "ground_truth_billing_m3": gt_billing,
            "billing_m3_val": int(gt_billing),
            "fraction_digit": int(gt_full[4]) if len(gt_full) >= 5 else 0,
            "type": "real_camera",
            "description": f"Camera ESP32-S3 thực tế: {gt_billing} m3 (chỉ số phụ {gt_full[4]})"
        })
        
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Suite 1 (Real Baseline): {len(manifest)} ảnh.")

def build_suite_2(masks, base_images):
    """
    Suite 2: Billing Digits Balanced (100 ảnh)
    Phân bố đồng đều các số 0-9 ở cả 4 ô đen (0000 - 9999 m3) dùng để tính tiền nước.
    """
    out_dir = os.path.join(ROOT_TEST_SUITES, "suite_2_billing_digits_balanced")
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    random.seed(42)
    
    # Tạo 100 số đảm bảo phân bố 0-9 đều trên từng cột trong 4 ô đen tính tiền
    digits_matrix = []
    for col in range(4):
        col_digits = [d for d in range(10)] * 10
        random.shuffle(col_digits)
        digits_matrix.append(col_digits)
        
    # Số đỏ thứ 5 phân bố đều 0-9
    red_col = [d for d in range(10)] * 10
    random.shuffle(red_col)
    
    for idx in range(1, 101):
        i = idx - 1
        billing_str = "".join(str(digits_matrix[col][i]) for col in range(4))
        full_str = f"{billing_str}{red_col[i]}"
        
        base_path = base_images[i % len(base_images)]
        img = generate_multiroll_meter(full_str, masks, base_path, roll_dict={})
        fname = f"meter_billing_bal_{idx:03d}_{billing_str}_{red_col[i]}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), img)
        
        manifest.append({
            "file": fname,
            "ground_truth_full": full_str,
            "ground_truth_billing_m3": billing_str,
            "billing_m3_val": int(billing_str),
            "fraction_digit": red_col[i],
            "type": "billing_balanced_static",
            "description": f"Chỉ số nước tính tiền: {billing_str} m3 (chữ số phụ: {red_col[i]})"
        })
        
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Suite 2 (Billing Digits Balanced): {len(manifest)} ảnh.")

def build_suite_3(masks, base_images):
    """
    Suite 3: Billing Jump Transitions & Floor Rule (50 ảnh)
    Kiểm tra cơ chế bước nhảy cơ học 9 -> 0 trên các bánh xe đen và
    nguyên tắc Làm tròn sàn (Floor Rule) bảo vệ quyền lợi người tiêu dùng:
    - Nhóm 1 (20 ảnh): Bước chuyển hàng chục đen (..19 -> ..20, ..49 -> ..50)
    - Nhóm 2 (15 ảnh): Bước chuyển hàng trăm đen (..099 -> ..100, ..599 -> ..600)
    - Nhóm 3 (15 ảnh): Bước chuyển hàng ngàn đen (..0999 -> ..1000, ..1999 -> ..2000)
    """
    out_dir = os.path.join(ROOT_TEST_SUITES, "suite_3_billing_jump_transitions")
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    random.seed(999)
    
    for idx in range(1, 51):
        base_path = base_images[(idx - 1) % len(base_images)]
        
        if idx <= 20:
            # Nhóm 1: Chuyển hàng chục đen (Bánh xe đen 3 đang là 9, bánh xe đen 2 đang chớm nhảy)
            prefix = f"{random.randint(0, 99):02d}"
            d_tens = random.randint(0, 8)
            billing_true = f"{prefix}{d_tens}9"
            red_digit = random.choice([8, 9, 0])
            full_str = f"{billing_true}{red_digit}"
            
            # Ô đen 3 ở mức 9 (đang chớm cuộn nhẹ sang 0), ô đen 2 hơi nhấp nhô
            # Quy tắc sàn: Ô đen 2 BẮT BUỘC phải đọc là d_tens, KHÔNG ĐƯỢC nhảy lên d_tens+1
            roll_dict = {3: 0.12, 2: 0.08}
            desc = f"Bước chuyển hàng chục đen: {billing_true} m3 (Ô 3 = 9, Ô 2 chớm nhấp nhô -> Tuân thủ Floor Rule đọc {d_tens})"
            
        elif idx <= 35:
            # Nhóm 2: Chuyển hàng trăm đen (..d99 -> ..(d+1)00)
            d_thousands = random.randint(0, 9)
            d_hundreds = random.randint(0, 8)
            billing_true = f"{d_thousands}{d_hundreds}99"
            red_digit = random.choice([8, 9, 0])
            full_str = f"{billing_true}{red_digit}"
            
            # Ô đen 3=9, ô đen 2=9, ô đen 1 đang chớm nhấp nhô
            roll_dict = {3: 0.14, 2: 0.12, 1: 0.09}
            desc = f"Bước chuyển hàng trăm đen: {billing_true} m3 (Ô 3,2 = 9, Ô 1 chớm nhấp nhô -> Tuân thủ Floor Rule đọc {d_hundreds})"
            
        else:
            # Nhóm 3: Chuyển hàng ngàn đen (d999 -> (d+1)000)
            d_thousands = random.randint(0, 8)
            billing_true = f"{d_thousands}999"
            red_digit = random.choice([8, 9, 0])
            full_str = f"{billing_true}{red_digit}"
            
            # 3 ô đen 999, ô đầu chớm nhấp nhô
            roll_dict = {3: 0.15, 2: 0.13, 1: 0.11, 0: 0.08}
            desc = f"Bước chuyển hàng ngàn đen: {billing_true} m3 (3 ô đen 999, Ô 0 chớm nhấp nhô -> Tuân thủ Floor Rule đọc {d_thousands})"
            
        img = generate_multiroll_meter(full_str, masks, base_path, roll_dict=roll_dict)
        fname = f"meter_jump_{idx:03d}_{billing_true}_{red_digit}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), img)
        
        manifest.append({
            "file": fname,
            "ground_truth_full": full_str,
            "ground_truth_billing_m3": billing_true,
            "billing_m3_val": int(billing_true),
            "fraction_digit": red_digit,
            "roll_dict": roll_dict,
            "type": "billing_jump_transition",
            "description": desc
        })
        
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Suite 3 (Billing Jump Transitions & Floor Rule): {len(manifest)} ảnh.")

def clean_old_suites():
    """Dọn dẹp các thư mục test cũ không thực tế để hệ sinh thái gọn gàng, rõ ràng."""
    old_folders = [
        "suite_2_balanced_digits",
        "suite_3_single_rolling_unit",
        "suite_4_cascade_rolling",
        "suite_5_environmental_stress"
    ]
    for folder in old_folders:
        p = os.path.join(ROOT_TEST_SUITES, folder)
        if os.path.exists(p):
            shutil.rmtree(p)
            print(f"[CLEANUP] Đã loại bỏ thư mục cũ: {p}")

def main():
    print("=" * 80)
    print("XÂY DỰNG 3 TẬP KIỂM THỬ ĐỘC LẬP CHUẨN NGHIỆP VỤ TÍNH TIỀN NƯỚC SAWACO")
    print("=" * 80)
    
    clean_old_suites()
    
    os.makedirs(ROOT_TEST_SUITES, exist_ok=True)
    masks = load_clean_masks()
    base_images = sorted(glob.glob(os.path.join(REAL_DATA_DIR, "*.jpg")))
    
    if not base_images:
        print("[ERROR] Không tìm thấy ảnh nền trong 'real_data_base'.")
        return
        
    print(f"[INFO] Tải thành công {len(masks)} mặt nạ chuẩn và {len(base_images)} ảnh canvas thực tế.")
    
    build_suite_1()
    build_suite_2(masks, base_images)
    build_suite_3(masks, base_images)
    
    print("\n" + "=" * 80)
    print("HOÀN TẤT! 3 TẬP KIỂM THỬ ĐÃ SẴN SÀNG TẠI 'data/test_suites/':")
    print("1. suite_1_real_baseline (23 ảnh thực địa)")
    print("2. suite_2_billing_digits_balanced (100 ảnh phủ đều 0000 - 9999 m3)")
    print("3. suite_3_billing_jump_transitions (50 ảnh bước nhảy 9->0 và làm tròn sàn Floor Rule)")
    print("=" * 80)

if __name__ == "__main__":
    main()
