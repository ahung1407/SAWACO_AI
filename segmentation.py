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
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5)
    
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
    
    digits = []
    
    # Nếu tìm ĐÚNG số lượng khung chữ số (ví dụ: 5)
    if len(kept_rects) == num_digits:
        print("[INFO] Đã nhận diện thành công các khung chữ số bằng Contours.")
        for i, (rx, ry, rw, rh) in enumerate(kept_rects):
            # Cắt lẹm vào trên/dưới một chút (khoảng 8%) để loại bỏ phần đỉnh/đáy của số bị lòi ra do đang lăn
            margin_y = int(rh * 0.08)
            # Cắt lẹm 2 bên trái phải một chút (khoảng 5%) để bỏ bóng râm của khung
            margin_x = int(rw * 0.05)
            
            # Đảm bảo không cắt lẹm quá đà
            if margin_y * 2 >= rh: margin_y = 0
            if margin_x * 2 >= rw: margin_x = 0
            
            # Lấy vùng ảnh dựa trên khung tìm được và cắt sâu vào trong
            box_img = img[ry+margin_y : ry+rh-margin_y, rx+margin_x : rx+rw-margin_x]
            
            # Pad to square
            ch, cw = box_img.shape[:2]
            size = max(ch, cw)
            pad_top = (size - ch) // 2
            pad_bottom = size - ch - pad_top
            pad_left = (size - cw) // 2
            pad_right = size - cw - pad_left
            
            digit_squared = cv2.copyMakeBorder(box_img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[240, 240, 240])
            
            # Thêm viền xung quanh (padding 15%) để số lọt thỏm vào giữa ảnh, giống chuẩn MNIST
            p = int(size * 0.15)
            digit_padded = cv2.copyMakeBorder(digit_squared, p, p, p, p, cv2.BORDER_CONSTANT, value=[240, 240, 240])
            
            digits.append(digit_padded)
            
        return digits
    else:
        print(f"[WARN] Tìm thấy {len(kept_rects)} khung, khác với yêu cầu ({num_digits}). Sử dụng phương pháp chia đều.")
        
    # --- PHƯƠNG PHÁP 2: CHIA ĐỀU (Fallback) ---
    # Cắt phần nhựa bên trái (như code cũ)
    trim_left = int(w * 0.056)
    img_fallback = img[:, trim_left:]
        
    h_f, w_f = img_fallback.shape[:2]
    box_w = w_f // num_digits
    digits = []
    
    for i in range(num_digits):
        bw = box_w
        bh = h_f
        
        margin_left = int(bw * 0.15)
        margin_right = int(bw * 0.15)
        margin_y = int(bh * 0.05)
        
        if i == num_digits - 1:
            margin_right = int(bw * 0.02)
        
        x = i * bw
        box_img = img_fallback[:, x:x+bw]
        
        if margin_left + margin_right >= bw:
            margin_left = 0
            margin_right = 0
        if margin_y * 2 >= bh:
            margin_y = 0
            
        digit_cropped = box_img[margin_y:bh-margin_y, margin_left:bw-margin_right]
        
        ch, cw = digit_cropped.shape[:2]
        size = max(ch, cw)
        pad_top = (size - ch) // 2
        pad_bottom = size - ch - pad_top
        pad_left = (size - cw) // 2
        pad_right = size - cw - pad_left
        
        digit_squared = cv2.copyMakeBorder(digit_cropped, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[240, 240, 240])
        digits.append(digit_squared)
        
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
