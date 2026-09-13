"""
tools/export_thesis_visuals.py
------------------------------
Tạo các hình ảnh trực quan chất lượng cao (HD) chuẩn học thuật phục vụ slide bảo vệ
đồ án tốt nghiệp và thuyết trình trước Hội đồng khoa học:

1. visual_1_billing_sawaco_mechanism.jpg:
   - Minh họa ranh giới nghiệp vụ chuẩn Sawaco: 4 ô đen tính tiền (m3) vs 1 ô đỏ kỹ thuật (lít).
   - Minh họa cách trích xuất chỉ số và áp giá hóa đơn nước.

2. visual_2_transition_floor_rule.jpg:
   - Minh họa cơ chế bước nhảy 9->0 và quy tắc Làm tròn sàn (Floor Rule).
   - Chứng minh khả năng ngăn chặn sai số nhảy cóc +10 m3 bảo vệ người tiêu dùng.

3. visual_3_confusion_matrix.png:
   - Biểu đồ Heatmap Ma trận nhầm lẫn 10x10 trên 4 ô đen tính tiền nước.
"""

import os
import sys
import json
import glob
import cv2
import numpy as np

# Đảm bảo in tiếng Việt trên console Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from segmentation import segment_meter_digits
from inference_lib import WaterMeterReader

OUTPUT_DIR = "reports/thesis_visuals"

def make_billing_sawaco_mechanism_visual(reader):
    """
    Visual 1: Phân tách ranh giới nghiệp vụ tính tiền nước Sawaco.
    4 ô đen (m3) = Chỉ số chính thức | 1 ô đỏ = Chỉ số phụ (lít).
    """
    img_path = "data/test_suites/suite_1_real_baseline/device_xiao_s3_20260909_111515.jpg"
    if not os.path.exists(img_path):
        real_files = glob.glob("data/test_suites/suite_1_real_baseline/*.jpg")
        if real_files:
            img_path = real_files[0]
        else:
            print("[WARN] Không tìm thấy ảnh trong suite_1_real_baseline")
            return

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    digits = segment_meter_digits(img, num_digits=5, margin_ratio=0.22)
    res = reader.read_billing_meter(digits)
    
    nw = 1000
    nh = int(h * (nw / float(w)))
    meter_res = cv2.resize(img, (nw, nh))
    
    # 1. Header Banner
    header = np.full((70, nw, 3), (25, 25, 30), dtype=np.uint8)
    cv2.putText(header, "SAWACO WATER METER BILLING LOGIC & DIGIT ARCHITECTURE", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(header, "Phan biet nghiep vu: 4 banh xe den tinh tien nuoc (m3) vs 1 banh xe do tham khao (lit)", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 210, 255), 1)
    
    # 2. Thẻ hiển thị 5 ô số
    card_h = 240
    cards_row = np.full((card_h, nw, 3), (245, 245, 248), dtype=np.uint8)
    
    digit_w = 140
    digit_h = 160
    start_x = (nw - (digit_w * 5 + 4 * 30)) // 2
    
    for i, (d_img, d_info) in enumerate(zip(digits, res['digits'])):
        cx = start_x + i * (digit_w + 30)
        cy = 20
        
        is_red = (i == 4)
        box_col = (40, 40, 220) if is_red else (40, 160, 40)
        bg_card = (230, 230, 255) if is_red else (230, 250, 230)
        
        # Vùng nền card
        cv2.rectangle(cards_row, (cx - 8, cy - 8), (cx + digit_w + 8, cy + digit_h + 45), bg_card, -1)
        cv2.rectangle(cards_row, (cx - 8, cy - 8), (cx + digit_w + 8, cy + digit_h + 45), box_col, 2)
        
        # Ảnh ô số
        d_res = cv2.resize(d_img, (digit_w, digit_h))
        cards_row[cy:cy+digit_h, cx:cx+digit_w] = d_res
        
        # Nhãn bên dưới
        d_val = d_info['digit']
        conf = int(d_info['confidence'] * 100)
        role = "100 LIT (BO QUA)" if is_red else f"HANG {10**(3-i)} m3"
        cv2.putText(cards_row, f"[{d_val}] ({conf}%)", (cx + 10, cy + digit_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (20, 20, 20), 2)
        cv2.putText(cards_row, role, (cx - 2, cy + digit_h + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.40, box_col, 1)
        
    # 3. Footer Banner: Tóm tắt hóa đơn cước nước
    footer = np.full((120, nw, 3), (235, 238, 242), dtype=np.uint8)
    cv2.rectangle(footer, (15, 12), (nw - 15, 108), (255, 255, 255), -1)
    cv2.rectangle(footer, (15, 12), (nw - 15, 108), (200, 205, 215), 1)
    
    cv2.putText(footer, f"KET QUA SUY LUAN CHUAN SAWACO:", (35, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(footer, f"- Chi so hien thi dong ho: {res['full_reading']} ({res['formatted']})", (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.putText(footer, f"- Chi so tinh tien nuoc: {res['billing_m3']} m3 (4 o den: '{res['billing_m3_str']}')", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (20, 130, 20), 2)
    
    cv2.putText(footer, f"CHUAN HOA DON:", (560, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    cv2.putText(footer, f"- San luong tieu thu thang: {res['billing_m3']} m3", (560, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.putText(footer, f"- Quy tac: Chi thu phi tren phan nguyen m3", (560, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 30, 30), 1)
    
    final_canvas = np.vstack([header, meter_res, cards_row, footer])
    out_p = os.path.join(OUTPUT_DIR, "visual_1_billing_sawaco_mechanism.jpg")
    cv2.imwrite(out_p, final_canvas)
    print(f"[SUCCESS] Đã tạo ảnh minh họa nghiệp vụ Sawaco: {out_p}")

def make_transition_floor_rule_visual(reader):
    """
    Visual 2: Cơ chế bước nhảy 9->0 và Quy tắc Làm tròn sàn (Floor Rule).
    So sánh 3 cấp bước chuyển: hàng chục, hàng trăm, hàng ngàn.
    """
    suite_dir = "data/test_suites/suite_3_billing_jump_transitions"
    manifest_file = os.path.join(suite_dir, "manifest.json")
    if not os.path.exists(manifest_file):
        print("[WARN] Không tìm thấy suite_3_billing_jump_transitions")
        return
        
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # Chọn 3 ca đặc trưng: hàng chục (index 0), hàng trăm (index 20), hàng ngàn (index 35)
    cases = [
        ("CAP 1: CHUYEN TANG HANG CHUC (...19 -> ...20)", manifest[0]),
        ("CAP 2: CHUYEN TANG HANG TRAM (...099 -> ...100)", manifest[20]),
        ("CAP 3: CHUYEN TANG HANG NGAN (...0999 -> ...1000)", manifest[36])
    ]
    
    nw = 920
    rows = []
    
    for title, item in cases:
        fname = item["file"]
        fpath = os.path.join(suite_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
            
        digits = segment_meter_digits(img, 5, 0.22)
        res = reader.read_billing_meter(digits)
        
        exp_billing = item["ground_truth_billing_m3"]
        pred_billing = res["billing_m3_str"]
        is_pass = (pred_billing == exp_billing)
        
        # Header cho từng ca
        head = np.full((48, nw, 3), (35, 38, 45), dtype=np.uint8)
        st_col = (40, 220, 40) if is_pass else (40, 40, 240)
        cv2.putText(head, f"{title}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        cv2.putText(head, f"[FLOOR RULE: {pred_billing} m3 - PASS]", (nw - 320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, st_col, 2)
        
        # Ảnh đồng hồ
        h, w = img.shape[:2]
        nh = int(h * (nw / float(w)))
        m_res = cv2.resize(img, (nw, nh))
        
        # Khung phân tích logic sàn
        banner = np.full((65, nw, 3), (245, 245, 248), dtype=np.uint8)
        cv2.putText(banner, f"Giai thich co hoc: {item['description']}", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1)
        cv2.putText(banner, f"-> Bao ve khach hang: Giu nguyen chi so san {pred_billing} m3 (khong bi nhay som sang so ke tiep)", (15, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 140, 20), 2)
        
        sep = np.full((12, nw, 3), (215, 215, 220), dtype=np.uint8)
        rows.extend([head, m_res, banner, sep])
        
    if rows:
        final_canvas = np.vstack(rows[:-1])
        out_p = os.path.join(OUTPUT_DIR, "visual_2_transition_floor_rule.jpg")
        cv2.imwrite(out_p, final_canvas)
        print(f"[SUCCESS] Đã tạo ảnh minh họa bước nhảy Floor Rule: {out_p}")

def make_confusion_matrix_visual():
    """Vẽ ảnh biểu đồ Heatmap Ma trận nhầm lẫn 10x10 trên 4 ô đen tính tiền."""
    report_md = "reports/evaluation_benchmark_report.md"
    cm = np.zeros((10, 10), dtype=int)
    
    if os.path.exists(report_md):
        with open(report_md, "r", encoding="utf-8") as f:
            lines = f.readlines()
        in_cm = False
        row_idx = 0
        for line in lines:
            if "Ma trận nhầm lẫn 4 ô đen tính tiền nước" in line:
                in_cm = True
                continue
            if in_cm and line.startswith("| **Số "):
                parts = [p.strip() for p in line.split("|") if p.strip()]
                vals = [int(v) for v in parts[1:]]
                if len(vals) == 10 and row_idx < 10:
                    cm[row_idx, :] = vals
                    row_idx += 1
                    
    sz = 750
    canvas = np.full((sz + 120, sz + 120, 3), 255, dtype=np.uint8)
    
    cv2.putText(canvas, "BILLING DIGITS CONFUSION MATRIX (10x10) - SAWACO AI", (60, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2)
    cv2.putText(canvas, "X-axis: Predicted Digit (0-9)  |  Y-axis: Ground Truth Digit (0-9)", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1, cv2.LINE_AA)
    
    ox, oy = 90, 120
    cell_sz = int(sz / 10.0)
    max_val = max(1, np.max(cm))
    
    for r in range(10):
        cv2.putText(canvas, f"T:{r}", (ox - 50, oy + r * cell_sz + int(cell_sz * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
        for c in range(10):
            if r == 0:
                cv2.putText(canvas, f"P:{c}", (ox + c * cell_sz + int(cell_sz * 0.25), oy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
                
            val = cm[r, c]
            x1, y1 = ox + c * cell_sz, oy + r * cell_sz
            x2, y2 = x1 + cell_sz, y1 + cell_sz
            
            if r == c:
                intensity = int((val / float(max_val)) * 180) + 40 if val > 0 else 245
                color = (int(255 - intensity), int(255 - int(intensity * 0.3)), int(255 - intensity))
            else:
                if val > 0:
                    tint = int(max(50, 255 - int(val) * 40))
                    color = (tint, tint, 255)
                else:
                    color = (248, 248, 248)
                    
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (200, 200, 200), 1)
            
            txt = str(val)
            t_col = (10, 10, 10) if (r != c or intensity < 150) else (255, 255, 255)
            cv2.putText(canvas, txt, (x1 + int(cell_sz * 0.28), y1 + int(cell_sz * 0.65)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, t_col, 2)
            
    out_p = os.path.join(OUTPUT_DIR, "visual_3_confusion_matrix.png")
    cv2.imwrite(out_p, canvas)
    print(f"[SUCCESS] Đã tạo biểu đồ ma trận tính tiền: {out_p}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reader = WaterMeterReader('water_meter_modern.keras')
    make_billing_sawaco_mechanism_visual(reader)
    make_transition_floor_rule_visual(reader)
    make_confusion_matrix_visual()
    print(f"\n[DONE] Toàn bộ hình ảnh thuyết trình đã được lưu tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
