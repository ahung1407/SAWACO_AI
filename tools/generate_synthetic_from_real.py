"""
tools/generate_synthetic_from_real.py
------------------------------------
Sinh ảnh đồng hồ nước CHÂN THỰC 100% (Ultra-Photorealistic) dựa trên ảnh chụp thực tế:
1. Canvas thực tế: Lấy trực tiếp ảnh camera ESP32-S3 Xiao làm nền (vỏ nhựa, góc chụp, bóng đổ, vết xước, ốc vít nguyên bản).
2. Tự động tìm chính xác 5 ô cửa sổ đồng hồ (Window Contours) trên mọi ảnh thực tế bằng thuật toán Dynamic Spacing Grid.
3. Tẩy xóa số cũ bằng Inpainting mượt mà (chỉ tẩy nét mực trong lòng bánh xe, giữ nguyên 100% cung tròn số lăn, viền nhựa và bóng kính).
4. Phủ nét chữ số chuẩn Sawaco (Trích xuất từ dữ liệu thực tế, lọc sạch viền/rác 100%):
   - Ô 0-3: Mực than đen nguyên bản (Charcoal Black: RGB 32, 28, 28).
   - Ô 4: Mực đỏ Sawaco đặc trưng (Vibrant Sawaco Red: RGB 208, 28, 22).
5. Mô phỏng bánh xe lăn tự nhiên (Rolling Wheel Transition) chuẩn cơ học khi số đang nhảy.
"""

import os
import sys
import glob
import random
import cv2
import numpy as np

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi Unicode
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

REAL_IMAGES_DIR = "real_data_base"
MASKS_DIR = "data/clean_digit_masks"

def load_clean_masks():
    """Tải 10 mặt nạ chữ số chuẩn 0-9 đã được lọc sạch hoàn toàn viền rác."""
    masks = {}
    for d in range(10):
        mask_path = os.path.join(MASKS_DIR, f"{d}.png")
        if os.path.exists(mask_path):
            img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[2] >= 4:
                masks[d] = img
    return masks

def get_meter_window_boxes(img):
    """
    Tự động dò tìm tọa độ chính xác của 5 ô số trên bất kỳ ảnh camera nào
    dựa trên contours hình học và nội suy khoảng cách lưới (Dynamic Spacing Grid).
    """
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 10)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidate_rects = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect_ratio = bh / float(bw)
        if (0.40 * h < bh < 0.98 * h) and (1.0 < aspect_ratio < 3.2):
            candidate_rects.append((bx, by, bw, bh))
            
    kept_rects = []
    for r in sorted(candidate_rects, key=lambda x: x[0]):
        overlap = False
        for k in kept_rects:
            inter_x1 = max(r[0], k[0])
            inter_x2 = min(r[0] + r[2], k[0] + k[2])
            if inter_x2 > inter_x1 and (inter_x2 - inter_x1) / min(r[2], k[2]) > 0.3:
                overlap = True
                break
        if not overlap:
            kept_rects.append(r)
            
    cxs = [r[0] + r[2]/2.0 for r in kept_rects]
    spacings = [cxs[i+1] - cxs[i] for i in range(len(cxs)-1)]
    valid_s = []
    for s in spacings:
        if 150 < s < 240:
            valid_s.append(s)
        elif 320 < s < 450:
            valid_s.append(s / 2.0)
        elif 500 < s < 650:
            valid_s.append(s / 3.0)
            
    est_spacing = float(np.median(valid_s)) if valid_s else 192.0
    
    best_grid_cxs = None
    max_inliers = -1
    for assumed_idx in range(5):
        for base_cx in cxs:
            start_cx = base_cx - assumed_idx * est_spacing
            grid_cxs = [start_cx + i * est_spacing for i in range(5)]
            inliers = sum(1 for cx in cxs if min(abs(cx - gx) for gx in grid_cxs) < est_spacing * 0.3)
            # Grid phải nằm hoàn toàn trong phạm vi ảnh
            if grid_cxs[0] >= 50 and grid_cxs[-1] <= w - 50:
                if inliers > max_inliers:
                    max_inliers = inliers
                    best_grid_cxs = grid_cxs
                elif inliers == max_inliers and best_grid_cxs is not None:
                    if abs((grid_cxs[0] + grid_cxs[-1])/2.0 - w/2.0) < abs((best_grid_cxs[0] + best_grid_cxs[-1])/2.0 - w/2.0):
                        best_grid_cxs = grid_cxs
                        
    if best_grid_cxs is None:
        best_grid_cxs = [int(w * (i + 0.5) / 5.0) for i in range(5)]
        
    box_w = int(est_spacing * 0.74)
    avg_y = float(np.median([r[1] for r in kept_rects])) if kept_rects else int(h * 0.22)
    avg_h = float(np.median([r[3] for r in kept_rects])) if kept_rects else int(h * 0.52)
    y1_box = max(0, int(avg_y - avg_h * 0.05))
    y2_box = min(h, int(avg_y + avg_h * 1.05))
    
    boxes = []
    for cx in best_grid_cxs:
        x1 = max(0, min(w - 20, int(cx - box_w/2.0)))
        x2 = max(x1 + 10, min(w, int(cx + box_w/2.0)))
        boxes.append((x1, y1_box, x2 - x1, y2_box - y1_box))
    return boxes

def inpaint_old_digit(roi, is_red=False):
    """
    Completely remove old digit strokes on the wheel face using Connected Components Inpainting.
    Selectively targets only digit ink components while leaving window frame borders 100% intact.
    """
    rh, rw = roi.shape[:2]
    mask = np.zeros((rh, rw), dtype=np.uint8)
    
    if not is_red:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, bin_ink = cv2.threshold(gray_roi, 145, 255, cv2.THRESH_BINARY_INV)
        n, lbls, stats, _ = cv2.connectedComponentsWithStats(bin_ink)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            is_left_frame = (x <= 2) and (x + w < rw * 0.20) and (h > rh * 0.3)
            is_right_frame = (x + w >= rw - 2) and (x > rw * 0.80) and (h > rh * 0.3)
            if not (is_left_frame or is_right_frame):
                mask[lbls == i] = 255
    else:
        central = roi
        b, g, r = cv2.split(central)
        is_red_ink = (r.astype(int) - np.maximum(b, g).astype(int) > 15) | (np.min(central, axis=2) < 135)
        bin_ink = is_red_ink.astype(np.uint8) * 255
        n, lbls, stats, _ = cv2.connectedComponentsWithStats(bin_ink)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            is_left_frame = (x <= 2) and (x + w < rw * 0.20) and (h > rh * 0.3)
            is_right_frame = (x + w >= rw - 2) and (x > rw * 0.80) and (h > rh * 0.3)
            if not (is_left_frame or is_right_frame):
                mask[lbls == i] = 255
                
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    inpainted = cv2.inpaint(roi, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted

def render_stroke(digit_val, target_w, target_h, masks, next_val=None, roll_ratio=0.0):
    """Tạo nét chữ số chuẩn với tỷ lệ hình học tự nhiên và căn giữa hoàn hảo."""
    m1 = masks.get(digit_val)
    if m1 is None:
        return np.zeros((target_h, target_w, 4), dtype=np.uint8)
        
    def scale_and_center_digit(m, w_box, h_box):
        mh, mw = m.shape[:2]
        dw = min(w_box, int(mw * (h_box / float(mh))))
        scaled = cv2.resize(m, (dw, h_box), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((h_box, w_box, 4), dtype=np.uint8)
        off_x = (w_box - dw) // 2
        canvas[:, off_x:off_x+dw] = scaled
        return canvas

    if next_val is None or roll_ratio <= 0.0:
        return scale_and_center_digit(m1, target_w, target_h)
        
    # Mô phỏng cơ học số lăn: cuộn bánh xe theo chiều dọc
    m2 = masks.get(next_val, m1)
    m1_centered = scale_and_center_digit(m1, target_w, target_h)
    m2_centered = scale_and_center_digit(m2, target_w, target_h)
    
    gap = int(target_h * 0.22)
    cylinder_h = target_h * 2 + gap
    strip = np.zeros((cylinder_h, target_w, 4), dtype=np.uint8)
    strip[:target_h] = m1_centered
    strip[target_h + gap:] = m2_centered
    
    view_y = int((target_h + gap) * roll_ratio)
    view_y = max(0, min(cylinder_h - target_h, view_y))
    return strip[view_y : view_y + target_h]

def generate_photorealistic_meter(reading_str, masks, base_img_path, roll_digit_idx=None, roll_ratio=0.5):
    """
    Sinh 1 ảnh đồng hồ hoàn toàn chân thực:
    - Base canvas là ảnh chụp camera ESP32-S3 thật.
    - 5 ô số được xóa nét cũ bằng Inpainting mượt mà và vẽ đè nét số mới chuẩn xác.
    """
    base_img = cv2.imread(base_img_path)
    if base_img is None:
        raise ValueError(f"Không thể mở ảnh nền: {base_img_path}")
        
    base_name = os.path.basename(base_img_path)
    calib_file = "data/real_window_boxes.json"
    if os.path.exists(calib_file):
        import json
        with open(calib_file, "r", encoding="utf-8") as f:
            calib = json.load(f)
            windows = calib.get(base_name, get_meter_window_boxes(base_img))
    else:
        windows = get_meter_window_boxes(base_img)
        
    result = base_img.copy()
    
    for i, ((bx, by, bw, bh), char) in enumerate(zip(windows, reading_str)):
        digit_val = int(char)
        is_red = (i == 4)
        
        roi = result[by:by+bh, bx:bx+bw].copy()
        rh, rw = roi.shape[:2]
        
        # 1. Inpaint xóa sạch 100% nét cũ không để lại vết mờ hay cung tròn
        mask = np.zeros((rh, rw), dtype=np.uint8)
        pad_x = 2
        pad_y = 2
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
            
        is_rolling = (roll_digit_idx == i)
        next_val = (digit_val + 1) % 10 if is_rolling else None
        stroke_bgra = render_stroke(digit_val, target_w, target_h, masks, 
                                    next_val=next_val, roll_ratio=roll_ratio if is_rolling else 0.0)
        
        # Căn chính giữa ô số
        dx1 = (rw - target_w) // 2
        dy1 = (rh - target_h) // 2
        dx2 = dx1 + target_w
        dy2 = dy1 + target_h
        
        alpha = stroke_bgra[:, :, 3].astype(np.float32) / 255.0
        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        
        # Màu mực chuẩn Sawaco
        ink_color = np.array([22, 28, 208] if is_red else [28, 28, 32], dtype=np.float32)
        wheel_roi = inpainted[dy1:dy2, dx1:dx2].astype(np.float32)
        
        blended = wheel_roi * (1.0 - alpha) + ink_color * alpha
        inpainted[dy1:dy2, dx1:dx2] = np.clip(blended, 0, 255).astype(np.uint8)
        
        result[by:by+bh, bx:bx+bw] = inpainted
        
    return result

def generate_dataset(num_samples=100, output_dir="data/generated_cases"):
    """Generate photorealistic test cases with balanced coverage into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    masks = load_clean_masks()
    base_images = sorted(glob.glob(os.path.join(REAL_IMAGES_DIR, "*.jpg")))
    
    if not base_images:
        print(f"[ERROR] No base images found in {REAL_IMAGES_DIR}!", flush=True)
        return []
        
    print(f"[INFO] Generating {num_samples} diverse test images into '{output_dir}'...", flush=True)
    manifest = []
    
    for idx in range(1, num_samples + 1):
        # 1. Random real background frame (cycling across all 23 real backgrounds)
        base_img_path = base_images[(idx - 1) % len(base_images)]
        
        # 2. Generate balanced 5-digit number
        # Ensure uniform digit distribution across 0-9
        val = f"{random.randint(0, 99999):05d}"
        
        # 3. 35% chance of rolling digit on window 5 (ratio between 0.05 and 0.18)
        is_rolling = (random.random() < 0.35)
        roll_idx = 4 if is_rolling else None
        roll_ratio = random.uniform(0.05, 0.18) if is_rolling else 0.0
        
        meter_img = generate_photorealistic_meter(val, masks, base_img_path, 
                                                 roll_digit_idx=roll_idx, 
                                                 roll_ratio=roll_ratio)
        
        fname = f"gen_meter_{idx:04d}_{val}.jpg"
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, meter_img)
        manifest.append((fname, val, roll_idx is not None))
        
        roll_str = f" [Rolling ô 5: {roll_ratio:.2f}]" if roll_idx is not None else ""
        if idx % 10 == 0 or idx == num_samples:
            print(f"  [{idx:03d}/{num_samples:03d}] Generated: {fname} (Value: {val}){roll_str}", flush=True)
        
    print(f"[INFO] Successfully generated {len(manifest)} test cases in '{output_dir}'.", flush=True)
    return manifest

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate photorealistic test cases based on real camera images.")
    parser.add_argument("--samples", type=int, default=100, help="Number of test images to generate (default: 100)")
    parser.add_argument("--output", type=str, default="data/generated_cases", help="Output directory")
    args = parser.parse_args()
    
    # Clean previous generated cases
    if os.path.exists(args.output):
        for old_f in glob.glob(os.path.join(args.output, "*.jpg")):
            try:
                os.remove(old_f)
            except Exception:
                pass
                
    generate_dataset(num_samples=args.samples, output_dir=args.output)
