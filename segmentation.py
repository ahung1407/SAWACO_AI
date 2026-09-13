import cv2
import numpy as np
import os

def _pad_to_square(img_gray):
    """Pad ảnh xám thành hình vuông kèm 15% viền trắng xung quanh"""
    ch_c, cw_c = img_gray.shape[:2]
    size = max(ch_c, cw_c)
    if size == 0:
        size = 1
    pad_t = (size - ch_c) // 2
    pad_b = size - ch_c - pad_t
    pad_l = (size - cw_c) // 2
    pad_r = size - cw_c - pad_l
    
    squared = cv2.copyMakeBorder(img_gray, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=255)
    p = int(size * 0.15)
    padded = cv2.copyMakeBorder(squared, p, p, p, p, cv2.BORDER_CONSTANT, value=255)
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)

def extract_clean_digit_ink(box_img):
    """
    [PREPROCESSING GATE - CONNECTED COMPONENTS]
    Loại bỏ 100% thanh viền nhựa đen bên hông và vết cuộn số lăn ở mép trên/dưới.
    Chỉ trích xuất duy nhất vùng nét mực (Ink Bounding Box) của chữ số trung tâm.
    Tự động hàn gắn (merge) các phần chữ số bị đứt gãy do lóa sáng đèn Flash.
    """
    if box_img is None or box_img.size == 0:
        return np.full((100, 100, 3), 255, dtype=np.uint8)
        
    bh, bw = box_img.shape[:2]
    if bh < 10 or bw < 10:
        return np.full((100, 100, 3), 255, dtype=np.uint8)
        
    # 1. Chuyển sang ảnh xám dùng np.min (chữ đỏ hay đen đều thành nét mực tối)
    gray = np.min(box_img, axis=2).astype(np.uint8) if len(box_img.shape) == 3 else box_img.copy()
    
    # 2. Nhị phân hóa Otsu: Nét mực = 255, Nền trắng = 0
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3. Phân tích Connected Components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, 8)
    if num_labels <= 1:
        return _pad_to_square(gray)
        
    # Lọc bỏ trước các thanh viền đen dạng cột đứng ở 2 mép
    # Một viền mép phải nằm gọn ở sát mép trái/phải, không được ăn sâu vào giữa
    non_border_lbls = []
    for lbl in range(1, num_labels):
        x, y, cw, ch, area = stats[lbl]
        is_left_border = (x <= bw * 0.12) and ((x + cw) <= bw * 0.30) and (ch >= bh * 0.25) and (ch / max(1, cw) >= 1.5)
        is_right_border = (x >= bw * 0.70) and ((x + cw) >= bw * 0.88) and (ch >= bh * 0.25) and (ch / max(1, cw) >= 1.5)
        if not (is_left_border or is_right_border) and area >= 50:
            non_border_lbls.append(lbl)
            
    if not non_border_lbls:
        return _pad_to_square(gray)
        
    # Tìm component chính (diện tích lớn nhất ở khu vực chữ số)
    main_lbl = max(non_border_lbls, key=lambda l: stats[l][4])
    mx, my, mw, mh, marea = stats[main_lbl]
    mcx, mcy = centroids[main_lbl]
    
    keep_lbls = [main_lbl]
    
    # Duyệt các component còn lại để phân biệt:
    # - Mảnh vỡ của cùng 1 chữ số (do lóa sáng): thẳng hàng theo trục X, khoảng cách gap_y nhỏ -> GIỮ LẠI
    # - Vết cuộn số lăn lửng lơ ở mép trên/dưới: cách xa > 25px -> LOẠI BỎ
    for lbl in non_border_lbls:
        if lbl == main_lbl:
            continue
        x, y, cw, ch, area = stats[lbl]
        cx, cy = centroids[lbl]
        
        if y >= my + mh:
            gap_y = y - (my + mh)
        elif y + ch <= my:
            gap_y = my - (y + ch)
        else:
            gap_y = 0
            
        align_x = abs(cx - mcx) < max(mw, cw) * 0.5
        total_span_y = max(my + mh, y + ch) - min(my, y)
        
        if align_x and gap_y <= 25 and total_span_y <= bh * 0.85:
            keep_lbls.append(lbl)
            
    clean_mask = np.zeros_like(gray, dtype=np.uint8)
    valid_boxes = []
    for lbl in keep_lbls:
        clean_mask[labels == lbl] = 255
        x, y, cw, ch, _ = stats[lbl]
        valid_boxes.append((x, y, x + cw, y + ch))
        
    min_x = max(0, min(b[0] for b in valid_boxes))
    min_y = max(0, min(b[1] for b in valid_boxes))
    max_x = min(bw, max(b[2] for b in valid_boxes))
    max_y = min(bh, max(b[3] for b in valid_boxes))
    
    # Dilate nhẹ mask để bảo toàn độ mượt nét chữ và hàn gắn vết lóa sáng mỏng
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated_mask = cv2.dilate(clean_mask, k, iterations=1)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    dilated_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, k_close)
    
    # Toàn bộ vùng ngoài nét mực biến thành màu trắng tinh (255)
    clean_gray = np.full_like(gray, 255)
    clean_gray[dilated_mask == 255] = gray[dilated_mask == 255]
    
    # Crop sát vùng nét chữ kèm lề an toàn 12%
    pad_w = int((max_x - min_x) * 0.12)
    pad_h = int((max_y - min_y) * 0.12)
    c_x1 = max(0, min_x - pad_w)
    c_y1 = max(0, min_y - pad_h)
    c_x2 = min(bw, max_x + pad_w)
    c_y2 = min(bh, max_y + pad_h)
    
    cropped = clean_gray[c_y1:c_y2, c_x1:c_x2]
    if cropped.size == 0:
        cropped = gray
        
    # [GEOMETRY GATE] Kiểm tra mật độ điểm ảnh để phát hiện ô bị che lấp / bùn đất
    _, binary_check = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    pixel_ratio = cv2.countNonZero(binary_check) / float(binary_check.size if binary_check.size > 0 else 1)
    if pixel_ratio < 0.05 or pixel_ratio > 0.65:
        return np.zeros((100, 100, 3), dtype=np.uint8)
        
    return _pad_to_square(cropped)

def segment_meter_digits(image_input, num_digits=5, margin_ratio=0.1):
    """
    Cắt một ảnh khung chứa dãy số thẳng ngang thành các ảnh chữ số riêng biệt.
    Sử dụng chiến lược mới: Nhận diện các khung hình chữ nhật chứa chữ số,
    nếu không tìm thấy đủ số lượng thì chuyển về phương pháp chia đều.
    
    Args:
        image_input (str hoặc numpy array): Đường dẫn đến ảnh HOẶC ảnh dưới dạng numpy array.
        num_digits (int): Số lượng chữ số có trong khung (mặc định 5).
        margin_ratio (float): Tỷ lệ lề cần cắt bỏ để loại bỏ khung nhựa giữa các số.
    
    Returns:
        list of numpy arrays: Danh sách các ảnh chữ số đã được cắt (RGB).
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError("Không thể đọc ảnh bằng OpenCV.")
    else:
        img = image_input
        
    h, w = img.shape[:2]
    
    # --- PHƯƠNG PHÁP 1: TÌM CONTOURS KHUNG HÌNH CHỮ NHẬT ---
    # Chuyển xám và làm rõ cạnh
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # [UPGRADE] Khử nhiễu ảnh trước khi Adaptive Threshold
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    
    # Adaptive threshold để làm nổi bật khung nhựa (thường có độ sáng khác với nền chứa số)
    # [NEW] Dùng block_size động phụ thuộc vào kích thước ảnh để bắt được các khung lớn/nhỏ
    block_size = (int(h / 4) // 2) * 2 + 1
    if block_size < 15:
        block_size = 15
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 5)
    
    # Dùng morphology để nối các nét bị đứt
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Thử RETR_TREE nếu EXTERNAL không bắt được
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
    valid_rects = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        
        # Điều kiện để là 1 khung chữ số:
        # 1. Chiều cao khung từ 30% đến 98% chiều cao ảnh (bắt trọn cả chữ số ngắn 140px lẫn khung dài)
        if 0.30 * h < bh < 0.98 * h:
            # 2. Tỷ lệ width/height từ 0.28 đến 0.95 (khung dọc)
            aspect_ratio = bw / float(bh)
            if 0.28 < aspect_ratio < 0.95:
                # 3. Chiều rộng tối thiểu >= 5% chiều rộng ảnh
                if bw >= w * 0.05:
                    valid_rects.append((x, y, bw, bh))
                
    # Lọc bỏ các khung trùng lặp: Ưu tiên khung có tỷ lệ width/height gần tỷ lệ chữ số chuẩn (~0.58) thay vì lấy khung cao to của vỏ
    valid_rects = sorted(valid_rects, key=lambda r: abs((r[2] / float(r[3])) - 0.58))
    kept_rects = []
    for r in valid_rects:
        rcx = r[0] + r[2]/2.0
        if not any(abs(rcx - (k[0] + k[2]/2.0)) < 110 for k in kept_rects):
            kept_rects.append(r)
            
    # Sắp xếp các khung từ trái sang phải
    kept_rects = sorted(kept_rects, key=lambda r: r[0])
    
    # --- SUY DIỄN LƯỚI TỌA ĐỘ BẰNG TÂM CHỮ SỐ (CENTER-BASED GRID INFERENCE) ---
    if len(kept_rects) == 0:
        print(f"[WARN] No contours found. Falling back to uniform grid.", flush=True)
        return _fallback_split(img, num_digits)
        
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
            
    # Khoảng cách chuẩn giữa 2 ô số trên camera ESP32-S3 Xiao luôn là ~192px
    est_spacing = float(np.median(valid_s)) if valid_s else (w * 0.178)
    
    if len(kept_rects) == num_digits and all(140 < s < 260 for s in spacings):
        best_grid_cxs = [r[0] + r[2]/2.0 for r in kept_rects]
        print("[INFO] Successfully detected digit bounding boxes using contours.", flush=True)
    elif len(kept_rects) == 1:
        # Tự động neo vị trí vật lý theo tọa độ X đã biết
        single_cx = cxs[0]
        # Ước lượng ô số thứ mấy dựa trên tọa độ X (mỗi ô cách nhau ~192px, ô 0 bắt đầu ~170px)
        assumed_idx = max(0, min(num_digits - 1, int(round((single_cx - w * 0.16) / est_spacing))))
        start_cx = single_cx - assumed_idx * est_spacing
        best_grid_cxs = [start_cx + i * est_spacing for i in range(num_digits)]
        print(f"[INFO] Single contour found at cx={single_cx:.1f} (anchored to index {assumed_idx}), synthesizing grid.", flush=True)

    else:
        best_grid_cxs = None
        max_inliers = -1
        
        # Giả định từng tâm tìm được ứng với vị trí thứ i trong 5 khung
        for assumed_idx in range(num_digits):
            for base_cx in cxs:
                start_cx = base_cx - assumed_idx * est_spacing
                grid_cxs = [start_cx + i * est_spacing for i in range(num_digits)]
                
                inliers = 0
                for cx in cxs:
                    dists = [abs(cx - gx) for gx in grid_cxs]
                    if min(dists) < est_spacing * 0.3:
                        inliers += 1
                        
                if inliers >= max_inliers:
                    if grid_cxs[0] > -est_spacing * 0.5 and grid_cxs[-1] < w + est_spacing * 0.5:
                        if inliers > max_inliers:
                            max_inliers = inliers
                            best_grid_cxs = grid_cxs
                        else: # inliers == max_inliers
                            current_center_offset = abs((grid_cxs[0] + grid_cxs[-1])/2 - w/2)
                            best_center_offset = abs((best_grid_cxs[0] + best_grid_cxs[-1])/2 - w/2) if best_grid_cxs else float('inf')
                            if current_center_offset < best_center_offset:
                                best_grid_cxs = grid_cxs
                    
    if not best_grid_cxs:
        print(f"[WARN] Unable to infer grid coordinates. Falling back to uniform grid.", flush=True)
        return _fallback_split(img, num_digits)
        
    if len(kept_rects) == num_digits:
        print("[INFO] Successfully detected digit bounding boxes using contours.", flush=True)
    else:
        print(f"[INFO] Contours detected {len(kept_rects)} boxes, successfully interpolated to {num_digits} boxes.", flush=True)
        
    digits = []
    # Lay box vua van khung chu so (72% spacing) de loai bo cac thanh vien nhua hai ben
    box_w = int(est_spacing * 0.72)
    
    debug_img = img.copy()
    
    if kept_rects:
        avg_y = float(np.median([r[1] for r in kept_rects]))
        avg_h = float(np.median([r[3] for r in kept_rects]))
        y1_box = max(0, int(avg_y - avg_h * 0.08))
        y2_box = min(h, int(avg_y + avg_h * 1.08))
    else:
        y1_box = int(h * 0.03)
        y2_box = int(h * 0.97)
        
    for i, cx in enumerate(best_grid_cxs):
        # Snap to physical window contour if detected nearby (prevents clipping wide digits)
        close_rects = [r for r in kept_rects if abs((r[0] + r[2]/2.0) - cx) < est_spacing * 0.20]
        if close_rects:
            best_r = min(close_rects, key=lambda r: abs((r[0] + r[2]/2.0) - cx))
            cx = best_r[0] + best_r[2] / 2.0
            
        x1 = int(cx - box_w/2)
        x2 = int(cx + box_w/2)
        y1 = y1_box
        y2 = y2_box
        
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        box_img = img[y1:y2, x1:x2]
        
        if box_img.size == 0 or x1 >= x2 or y1 >= y2:
            print(f"[WARN] Box out of bounds at x1={x1}, x2={x2}. Creating blank pad.", flush=True)
            box_img = np.full((50, 50, 3), 255, dtype=np.uint8)

        # [PREPROCESSING GATE] Gọt sạch viền nhựa đen và vết cuộn số
        is_red = (i == num_digits - 1)
        if is_red and len(box_img.shape) == 3:
            b_c, g_c, r_c = box_img[:,:,0], box_img[:,:,1], box_img[:,:,2]
            # Nền nhựa đen trung tính quanh ô số đỏ: R, G, B thấp và chênh lệch màu nhỏ
            is_black_frame = (r_c < 110) & (g_c < 110) & (b_c < 110) & (np.abs(r_c.astype(int) - g_c.astype(int)) < 35)
            box_img = box_img.copy()
            box_img[is_black_frame] = [255, 255, 255]

        clean_digit = extract_clean_digit_ink(box_img)
        digits.append(clean_digit)
        
    # Lưu ảnh debug 
    cv2.imwrite("debug_final_boxes.jpg", debug_img)
    return digits

def _fallback_split(img, num_digits):
    h, w = img.shape[:2]
    
    # 1. Tạo ảnh nhị phân để phân biệt nền sáng và chữ tối
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # Ảnh đồng hồ thường nền trắng/xanh sáng, chữ đen. Adaptive Threshold (INV) -> chữ trắng, nền đen
    block_size = (int(h / 4) // 2) * 2 + 1
    if block_size < 15:
        block_size = 15
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 5)
    
    # 2. Tính mật độ điểm ảnh theo chiều dọc (Vertical Projection Profile)
    # col_sums sẽ cao ở chỗ có nét chữ, thấp (Gần 0) ở chỗ khoảng trắng giữa các chữ
    col_sums = np.sum(thresh, axis=0)
    
    # 3. Tìm các đường cắt (valleys)
    # Thay vì chia đều, ta tìm điểm có ít mực nhất (local minimum) xung quanh các vị trí chia đều dự kiến
    expected_cuts = [int(w * i / num_digits) for i in range(1, num_digits)]
    search_window = int(w / num_digits * 0.4) # Cho phép xê dịch +/- 40% của 1 ô
    
    actual_cuts = [0]
    for xc in expected_cuts:
        left_bound = max(0, xc - search_window)
        right_bound = min(w, xc + search_window)
        
        # Tìm index có tổng pixel nhỏ nhất trong khoảng này
        window_sums = col_sums[left_bound:right_bound]
        if len(window_sums) > 0:
            local_min_idx = np.argmin(window_sums)
            best_cut = left_bound + local_min_idx
        else:
            best_cut = xc
            
        actual_cuts.append(best_cut)
    
    actual_cuts.append(w)
    
    # [NEW] Vẽ ảnh debug cho fallback
    debug_img = img.copy()
    
    digits = []
    for i in range(num_digits):
        x_start = actual_cuts[i]
        x_end = actual_cuts[i+1]
        
        # Vẽ khung cắt (dọc toàn màn hình nhưng chừa lề y)
        margin_y = int(h * 0.05)
        cv2.rectangle(debug_img, (x_start, margin_y), (x_end, h-margin_y), (0, 165, 255), 2) # Màu cam
        
        box_img = img[margin_y:h-margin_y, x_start:x_end]
        
        # [PREPROCESSING GATE] Gọt sạch viền nhựa đen và vết cuộn số
        clean_digit = extract_clean_digit_ink(box_img)
        digits.append(clean_digit)
        
    # Lưu ảnh debug fallback
    cv2.imwrite("debug_final_boxes.jpg", debug_img)
    return digits

if __name__ == "__main__":
    test_img = "test_meter.jpg"
    if os.path.exists(test_img):
        print(f"Segmenting {test_img}...", flush=True)
        digits = segment_meter_digits(test_img, num_digits=5, margin_ratio=0.15)
        
        for i, d in enumerate(digits):
            cv2.imwrite(f"digit_part_{i}.jpg", d)
            print(f"Saved digit_part_{i}.jpg (Shape: {d.shape})", flush=True)
    else:
        print("Create test_meter.jpg to test this module.", flush=True)
