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

# =========================================================================
# QUY CHUẨN PHÂN CHIA DỮ LIỆU KHOA HỌC: HOLD-OUT TEST SET (70% TRAIN - 30% TEST)
# 16 ảnh Train (để fine-tune) và 7 ảnh Test (cô lập 100%, chưa từng tham gia train)
# =========================================================================
TRAIN_REAL_FILES = [
    "device_xiao_s3_20260909_094142.jpg",
    "device_xiao_s3_20260909_095354.jpg",
    "device_xiao_s3_20260909_100525.jpg",
    "device_xiao_s3_20260909_101854.jpg",
    "device_xiao_s3_20260909_102216.jpg",
    "device_xiao_s3_20260909_102432.jpg",
    "device_xiao_s3_20260909_102638.jpg",
    "device_xiao_s3_20260909_102856.jpg",
    "device_xiao_s3_20260909_103034.jpg",
    "device_xiao_s3_20260909_103222.jpg",
    "device_xiao_s3_20260909_103920.jpg",
    "device_xiao_s3_20260909_104204.jpg",
    "device_xiao_s3_20260909_104558.jpg",
    "device_xiao_s3_20260909_104858.jpg",
    "device_xiao_s3_20260909_105206.jpg",
    "device_xiao_s3_20260909_105530.jpg"
]

TEST_REAL_FILES = [
    "device_xiao_s3_20260909_105724.jpg",
    "device_xiao_s3_20260909_105930.jpg",
    "device_xiao_s3_20260909_110142.jpg",
    "device_xiao_s3_20260909_110538.jpg",
    "device_xiao_s3_20260909_111235.jpg",
    "device_xiao_s3_20260909_111515.jpg",
    "device_xiao_s3_20260909_111955.jpg"
]

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
        
        # Thêm độ rơ cơ khí thực tế (Mechanical Jitter / Misalignment: ±2 px)
        jitter_x = random.randint(-2, 2)
        jitter_y = random.randint(-2, 2)
        dx1 = max(0, min(rw - target_w, (rw - target_w) // 2 + jitter_x))
        dy1 = max(0, min(rh - target_h, (rh - target_h) // 2 + jitter_y))
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
    """
    Suite 1: Real Baseline Hold-out (7 ảnh chụp thực tế ESP32-S3 HOÀN TOÀN ĐỘC LẬP).
    Mô hình chưa từng nhìn thấy bất kỳ ô số nào của 7 ảnh này trong lúc train/fine-tune.
    """
    out_dir = os.path.join(ROOT_TEST_SUITES, "suite_1_real_baseline")
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    
    for fname in TEST_REAL_FILES:
        path = os.path.join(REAL_DATA_DIR, fname)
        if not os.path.exists(path):
            continue
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
            "type": "real_camera_unseen_holdout",
            "description": f"Camera ESP32-S3 thực tế (Hold-out): {gt_billing} m3 (chỉ số phụ {gt_full[4]})"
        })
        
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Suite 1 (Real Baseline Hold-out): {len(manifest)} ảnh (100% Unseen Data).")

def apply_realistic_meter_environment(img, windows):
    """
    Mô phỏng chân thực các điều kiện bất lợi ngoài hiện trường thực tế của Sawaco:
    1. Nhiễu hạt cảm biến CMOS OV2640 (ESP32-S3 sensor noise).
    2. Đọng sương / mờ hơi nước trên mặt kính (Moisture / Condensation).
    3. Vết lóa sáng phản quang từ đèn Flash / ánh nắng (Glare reflection).
    4. Biến thiên ánh sáng / tương phản (Ambient lighting variation).
    """
    res = img.astype(np.float32)
    h, w = img.shape[:2]
    
    # 1. Biến thiên tương phản & độ sáng nhẹ (ánh sáng môi trường thực tế)
    contrast = random.uniform(0.90, 1.05)
    brightness = random.uniform(-8.0, 8.0)
    res = np.clip(res * contrast + brightness, 0, 255)
    
    # 2. Nhiễu hạt cảm biến CMOS ESP32-S3 (Gaussian noise)
    noise_sigma = random.uniform(2.0, 5.0)
    noise = np.random.normal(0, noise_sigma, res.shape).astype(np.float32)
    res = np.clip(res + noise, 0, 255)
    
    res = res.astype(np.uint8)
    
    # 3. 25% tỷ lệ xuất hiện vết chói sáng phản quang trên mặt kính (Flash / Sun Glare)
    if random.random() < 0.25 and windows:
        target_win = random.choice(windows)
        gx = target_win[0] + target_win[2] // 2 + random.randint(-10, 10)
        gy = target_win[1] + target_win[3] // 2 + random.randint(-10, 10)
        radius = random.randint(20, 40)
        
        glare_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(glare_mask, (gx, gy), radius, 1.0, -1)
        glare_mask = cv2.GaussianBlur(glare_mask, (25, 25), 9)
        
        glare_intensity = random.uniform(30.0, 55.0)
        for c in range(3):
            res[:, :, c] = np.clip(res[:, :, c].astype(np.float32) + glare_mask * glare_intensity, 0, 255).astype(np.uint8)
            
    # 4. 20% tỷ lệ bị mờ sương / đọng ẩm (Mild Condensation / Defocus)
    if random.random() < 0.20:
        res = cv2.GaussianBlur(res, (3, 3), 0.7)
        
    return res

def build_suite_2(masks, base_images):
    """
    Suite 2: Billing Digits Balanced Under Field Stress (100 ảnh)
    Phân bố đồng đều các số 0-9 ở cả 4 ô đen (0000 - 9999 m3) kết hợp các điều kiện
    môi trường khắc nghiệt thực tế: nhiễu hạt cảm biến, chói lóa đèn flash, mờ hơi nước.
    """
    out_dir = os.path.join(ROOT_TEST_SUITES, "suite_2_billing_digits_balanced")
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    random.seed(42)
    np.random.seed(42)
    
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
        
        # Áp dụng các điều kiện bất lợi thực tế ngoài hiện trường
        base_name = os.path.basename(base_path)
        calib_file = "data/real_window_boxes.json"
        if os.path.exists(calib_file):
            with open(calib_file, "r", encoding="utf-8") as f:
                calib = json.load(f)
                windows = calib.get(base_name, get_meter_window_boxes(img))
        else:
            windows = get_meter_window_boxes(img)
            
        img = apply_realistic_meter_environment(img, windows)
        
        fname = f"meter_billing_bal_{idx:03d}_{billing_str}_{red_col[i]}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), img)
        
        manifest.append({
            "file": fname,
            "ground_truth_full": full_str,
            "ground_truth_billing_m3": billing_str,
            "billing_m3_val": int(billing_str),
            "fraction_digit": red_col[i],
            "type": "billing_balanced_field_stress",
            "description": f"Chỉ số nước tính tiền: {billing_str} m3 (chữ số phụ: {red_col[i]}) - Môi trường thực địa (nhiễu/chói/sương)"
        })
        
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[SUCCESS] Suite 2 (Billing Digits Balanced): {len(manifest)} ảnh.")

def build_suite_3(masks, base_images):
    """
    Suite 3: Billing Jump Transitions & Floor Rule (50 ảnh)
    Tuân thủ 100% nguyên lý động lực học liên kết bánh răng (Geneva / Pinion Mechanism):
    - Một bánh xe bậc cao CHỈ CÓ THỂ quay lơ lửng khi TẤT CẢ các bánh xe bên phải nó
      đang ở chữ số 9 và trong tiến trình chuyển tiếp 9 -> 0!
    - Nhóm 1 (25 ảnh): Chuyển hàng đơn vị m3 (Ô 3 lơ lửng d -> d+1, ô đỏ 4 ở mốc 9 -> 0)
    - Nhóm 2 (15 ảnh): Chuyển hàng chục m3 (Ô 2 lơ lửng d -> d+1, ô 3 là 9 lăn 9->0, ô đỏ 4 lăn 9->0)
    - Nhóm 3 (10 ảnh): Chuyển hàng trăm / hàng ngàn m3 (Ô 1 hoặc 0 lơ lửng, toàn bộ bên phải là 9 lăn 9->0)
    """
    out_dir = os.path.join(ROOT_TEST_SUITES, "suite_3_billing_jump_transitions")
    os.makedirs(out_dir, exist_ok=True)
    manifest = []
    random.seed(999)
    
    roll_levels = [0.28, 0.38, 0.48, 0.58, 0.68]
    
    for idx in range(1, 51):
        base_path = base_images[(idx - 1) % len(base_images)]
        r_level = roll_levels[(idx - 1) % len(roll_levels)]
        
        if idx <= 25:
            # Nhóm 1 (25 ảnh): Chuyển hàng đơn vị đen (m3).
            # Ô 0, 1, 2 đứng yên.
            # Ô 3 lơ lửng giữa d_unit và d_unit+1.
            # Ô đỏ 4 BẮT BUỘC là số 9 đang chuyển tiếp sang 0 (kéo theo ô 3 nhích lên).
            d_thou = random.randint(0, 9)
            d_hund = random.randint(0, 9)
            d_tens = random.randint(0, 9)
            d_unit = random.randint(0, 8)  # chuyển từ d_unit -> d_unit + 1
            
            billing_true = f"{d_thou}{d_hund}{d_tens}{d_unit}"
            red_digit = 9  # Số đỏ đang ở 9 chuẩn bị qua 0
            full_str = f"{billing_true}{red_digit}"
            
            # Ô 3 quay r_level, ô đỏ 4 quay tỉ lệ cao 0.70 - 0.90 (gần chạm 0)
            red_roll = min(0.92, 0.65 + 0.35 * r_level)
            roll_dict = {3: r_level, 4: red_roll}
            desc = (f"Chuyển hàng đơn vị m3: {billing_true} m3 (Ô 3 quay {int(r_level*100)}% "
                    f"giữa {d_unit} và {d_unit+1}; Ô đỏ kế bên đang là 9 lăn {int(red_roll*100)}% sang 0)")
            
        elif idx <= 40:
            # Nhóm 2 (15 ảnh): Chuyển hàng chục đen (10 m3).
            # Ô 0, 1 đứng yên.
            # Ô 2 lơ lửng giữa d_tens và d_tens+1.
            # Ô 3 BẮT BUỘC là số 9 (đang lăn 9 -> 0).
            # Ô đỏ 4 BẮT BUỘC là số 9 (đang lăn 9 -> 0).
            d_thou = random.randint(0, 9)
            d_hund = random.randint(0, 9)
            d_tens = random.randint(0, 8)  # chuyển từ d_tens -> d_tens + 1
            d_unit = 9
            
            billing_true = f"{d_thou}{d_hund}{d_tens}{d_unit}"
            red_digit = 9
            full_str = f"{billing_true}{red_digit}"
            
            unit_roll = min(0.85, 0.40 + 0.50 * r_level)
            red_roll = min(0.95, 0.75 + 0.20 * r_level)
            roll_dict = {2: r_level, 3: unit_roll, 4: red_roll}
            desc = (f"Chuyển hàng chục m3: {billing_true} m3 (Ô 2 quay {int(r_level*100)}% giữa {d_tens} và {d_tens+1}; "
                    f"CÁC Ô BÊN PHẢI BẮT BUỘC: Ô 3 là 9 lăn sang 0, Ô đỏ 4 là 9 lăn sang 0)")
            
        else:
            # Nhóm 3 (10 ảnh): Chuyển hàng trăm đen (100 m3) hoặc hàng ngàn (1000 m3).
            # Ô 1 lơ lửng giữa d_hund và d_hund+1.
            # TẤT CẢ các ô bên phải (ô 2, ô 3, ô 4) BẮT BUỘC đều là 9 và đang cùng cascade sang 0!
            d_thou = random.randint(0, 8)
            d_hund = random.randint(0, 8)
            billing_true = f"{d_thou}{d_hund}99"
            red_digit = 9
            full_str = f"{billing_true}{red_digit}"
            
            tens_roll = min(0.75, 0.35 + 0.45 * r_level)
            unit_roll = min(0.88, 0.55 + 0.35 * r_level)
            red_roll = min(0.96, 0.80 + 0.16 * r_level)
            roll_dict = {1: r_level, 2: tens_roll, 3: unit_roll, 4: red_roll}
            desc = (f"Chuyển hàng trăm m3: {billing_true} m3 (Ô 1 quay {int(r_level*100)}% giữa {d_hund} và {d_hund+1}; "
                    f"TOÀN BỘ CÁC Ô BÊN PHẢI ĐỀU LÀ 9 ĐANG CHUYỂN SANG 0: Ô 2, Ô 3, Ô đỏ 4)")
            
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
    print(f"[SUCCESS] Suite 3 (Billing Jump Transitions & Floor Rule - Cơ học chuẩn 100%): {len(manifest)} ảnh.")

def clean_old_suites():
    """Xóa sạch hoàn toàn toàn bộ thư mục test suites cũ để sinh lại từ đầu 100% sạch sẽ."""
    if os.path.exists(ROOT_TEST_SUITES):
        shutil.rmtree(ROOT_TEST_SUITES)
        print(f"[CLEANUP] Đã xóa sạch toàn bộ thư mục kiểm thử cũ: {ROOT_TEST_SUITES}")
    os.makedirs(ROOT_TEST_SUITES, exist_ok=True)

def main():
    print("=" * 80)
    print("XÂY DỰNG 3 TẬP KIỂM THỬ ĐỘC LẬP CHUẨN NGHIỆP VỤ TÍNH TIỀN NƯỚC SAWACO")
    print("=" * 80)
    
    clean_old_suites()
    
    os.makedirs(ROOT_TEST_SUITES, exist_ok=True)
    masks = load_clean_masks()
    # CHỈ DÙNG 7 ẢNH THUỘC TẬP TEST ĐỘC LẬP LÀM CANVAS (CẤM DÙNG 16 ẢNH TRAIN)
    test_base_images = [os.path.join(REAL_DATA_DIR, f) for f in TEST_REAL_FILES if os.path.exists(os.path.join(REAL_DATA_DIR, f))]
    
    if not test_base_images:
        print("[ERROR] Không tìm thấy ảnh nền test trong 'real_data_base'.")
        return
        
    print(f"[INFO] Tải thành công {len(masks)} mặt nạ chuẩn và {len(test_base_images)} ảnh canvas TEST ĐỘC LẬP (Hold-out).")
    
    build_suite_1()
    build_suite_2(masks, test_base_images)
    build_suite_3(masks, test_base_images)
    
    print("\n" + "=" * 80)
    print("HOÀN TẤT! 3 TẬP KIỂM THỬ ĐÃ SẴN SÀNG TẠI 'data/test_suites/':")
    print(f"1. suite_1_real_baseline ({len(TEST_REAL_FILES)} ảnh thực địa Hold-out chưa từng thấy)")
    print("2. suite_2_billing_digits_balanced (100 ảnh sinh trên nền TEST độc lập)")
    print("3. suite_3_billing_jump_transitions (50 ảnh số lơ lửng sinh trên nền TEST độc lập)")
    print("=" * 80)

if __name__ == "__main__":
    main()
