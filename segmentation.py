import cv2
import numpy as np
import os

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
        # 1. Chiều cao khung thường chiếm từ 40% đến 95% chiều cao ảnh
        if 0.4 * h < bh < 0.98 * h:
            # 2. Tỷ lệ width/height thường từ 0.3 đến 0.8 (khung dọc)
            aspect_ratio = bw / float(bh)
            if 0.25 < aspect_ratio < 0.9:
                valid_rects.append((x, y, bw, bh))
                
    # Lọc bỏ các khung trùng lặp (Overlap)
    valid_rects = sorted(valid_rects, key=lambda r: r[2]*r[3], reverse=True) # Sắp xếp theo diện tích giảm dần
    kept_rects = []
    for r in valid_rects:
        overlap = False
        for k in kept_rects:
            # Tính độ giao nhau theo trục X
            inter_x1 = max(r[0], k[0])
            inter_x2 = min(r[0]+r[2], k[0]+k[2])
            if inter_x2 > inter_x1:
                inter_w = inter_x2 - inter_x1
                min_w = min(r[2], k[2])
                if inter_w / min_w > 0.3: # Giao nhau trên 30% chiều rộng thì coi là trùng
                    overlap = True
                    break
        if not overlap:
            kept_rects.append(r)
            
    # Sắp xếp các khung từ trái sang phải
    kept_rects = sorted(kept_rects, key=lambda r: r[0])
    
    # --- SUY DIỄN LƯỚI TỌA ĐỘ BẰNG TÂM CHỮ SỐ (CENTER-BASED GRID INFERENCE) ---
    if len(kept_rects) < 2:
        print(f"[WARN] Chi tim thay {len(kept_rects)} khung bang Contours. Khong the noi suy luoi. Su dung phuong phap chia deu.")
        return _fallback_split(img, num_digits)
        
    cxs = [r[0] + r[2]/2.0 for r in kept_rects]
    spacings = [cxs[i+1] - cxs[i] for i in range(len(cxs)-1)]
    if spacings:
        # Lọc các khoảng cách hợp lệ (ví dụ > 10% chiều rộng ảnh)
        valid_s = [s for s in spacings if s > w * 0.1]
        est_spacing = min(valid_s) if valid_s else w / float(num_digits)
    else:
        est_spacing = w / float(num_digits)
        
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
                    
            # [NEW] Nếu số inliers bằng nhau (thường xảy ra khi chỉ có 1 contour),
            # ta ưu tiên cái grid nào nằm cân đối nhất trong ảnh (ít bị lệch ra ngoài nhất)
            if inliers >= max_inliers:
                if grid_cxs[0] > -est_spacing * 0.5 and grid_cxs[-1] < w + est_spacing * 0.5:
                    if inliers > max_inliers:
                        max_inliers = inliers
                        best_grid_cxs = grid_cxs
                    else: # inliers == max_inliers
                        # Chọn grid có tâm tổng thể gần giữa ảnh nhất
                        current_center_offset = abs((grid_cxs[0] + grid_cxs[-1])/2 - w/2)
                        best_center_offset = abs((best_grid_cxs[0] + best_grid_cxs[-1])/2 - w/2) if best_grid_cxs else float('inf')
                        if current_center_offset < best_center_offset:
                            best_grid_cxs = grid_cxs
                    
    if not best_grid_cxs:
        print(f"[WARN] Khong the suy dien luoi toa do. Su dung phuong phap chia deu.")
        return _fallback_split(img, num_digits)
        
    if len(kept_rects) == num_digits:
        print("[INFO] Da nhan dien hoan hao cac khung chu so bang Contours.")
    else:
        print(f"[INFO] Contours tim thay {len(kept_rects)} khung, da tu dong noi suy thanh {num_digits} khung thanh cong.")
        
    digits = []
    box_w = int(est_spacing * 0.85)
    
    # [NEW] Vẽ ảnh debug cho phương pháp lưới
    debug_img = img.copy()
    
    for i, cx in enumerate(best_grid_cxs):
        x1 = int(cx - box_w/2)
        x2 = int(cx + box_w/2)
        y1 = int(h * 0.05)  # Bỏ 5% lề trên
        y2 = int(h * 0.95)  # Bỏ 5% lề dưới
        
        # Đảm bảo toạ độ cắt nằm trong ảnh
        x1 = max(0, min(w, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1))
        y2 = max(0, min(h, y2))
        
        # Vẽ khung màu xanh lá
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        box_img = img[y1:y2, x1:x2]
        
        # [NEW] Safety check to prevent empty array crash if the grid falls completely outside the image
        if box_img.size == 0 or x1 >= x2 or y1 >= y2:
            print(f"[WARN] Box out of bounds at x1={x1}, x2={x2}. Creating blank pad.")
            box_img = np.full((y2-y1 if y2>y1 else 50, x2-x1 if x2>x1 else 50, 3), 255, dtype=np.uint8)
            
        # [NEW] Cân bằng sáng toàn cục (Normalize) để nền chuyển thành trắng (255) và số thành đen (0)
        # Giúp triệt tiêu nền xám/xanh trước khi đưa vào CNN
        box_gray = np.min(box_img, axis=2).astype(np.uint8)
        box_norm = cv2.normalize(box_gray, None, 0, 255, cv2.NORM_MINMAX)
        box_img_bgr = cv2.cvtColor(box_norm, cv2.COLOR_GRAY2BGR)
        
        # Pad to square
        ch, cw = box_img_bgr.shape[:2]
        size = max(ch, cw)
        if size == 0:
            size = 1
        pad_top = (size - ch) // 2
        pad_bottom = size - ch - pad_top
        pad_left = (size - cw) // 2
        pad_right = size - cw - pad_left
        
        # Dùng màu trắng (255) để pad thay vì màu xám (240)
        digit_squared = cv2.copyMakeBorder(box_img_bgr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # Thêm viền xung quanh (padding 15%)
        p = int(size * 0.15)
        digit_padded = cv2.copyMakeBorder(digit_squared, p, p, p, p, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        digits.append(digit_padded)
        
    # Lưu ảnh debug 
    cv2.imwrite("debug_final_boxes.jpg", debug_img)
    return digits

def _fallback_split(img, num_digits):
    h, w = img.shape[:2]
    
    # 1. Tạo ảnh nhị phân để phân biệt nền sáng và chữ tối
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        
        # [NEW] Normalize ảnh giống như phương pháp contours để khử nhiễu sáng
        if box_img.size > 0:
            box_gray = np.min(box_img, axis=2).astype(np.uint8)
            box_norm = cv2.normalize(box_gray, None, 0, 255, cv2.NORM_MINMAX)
            digit_cropped_bgr = cv2.cvtColor(box_norm, cv2.COLOR_GRAY2BGR)
        else:
            digit_cropped_bgr = np.full((10, 10, 3), 255, dtype=np.uint8)
            
        ch, cw = digit_cropped_bgr.shape[:2]
        size = max(ch, cw)
        if size == 0: size = 1
        pad_top = (size - ch) // 2
        pad_bottom = size - ch - pad_top
        pad_left = (size - cw) // 2
        pad_right = size - cw - pad_left
        
        digit_squared = cv2.copyMakeBorder(digit_cropped_bgr, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        p = int(size * 0.15)
        digit_padded = cv2.copyMakeBorder(digit_squared, p, p, p, p, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        digits.append(digit_padded)
        
    # Lưu ảnh debug fallback
    cv2.imwrite("debug_final_boxes.jpg", debug_img)
    return digits

if __name__ == "__main__":
    test_img = "test_meter.jpg"
    if os.path.exists(test_img):
        print(f"Đang phân tách {test_img}...")
        digits = segment_meter_digits(test_img, num_digits=5, margin_ratio=0.15)
        
        for i, d in enumerate(digits):
            cv2.imwrite(f"digit_part_{i}.jpg", d)
            print(f"Đã lưu digit_part_{i}.jpg (Kích thước: {d.shape})")
    else:
        print("Tạo file test_meter.jpg để chạy thử module này.")
