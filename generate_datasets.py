import cv2
import numpy as np
import os
import shutil
import random

def ensure_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def draw_digit_on_layer(digit, font, scale, thickness, text_color, is_light_bg, h, w):
    """Hàm phụ trợ để vẽ chữ số lên một mặt phẳng (layer) trống"""
    layer = np.zeros((h, w, 3), dtype=np.uint8) if not is_light_bg else np.full((h, w, 3), 255, dtype=np.uint8)
    text_size = cv2.getTextSize(str(digit), font, scale, thickness)[0]
    
    # Canh giữa chữ số theo chiều ngang, chiều dọc tựa trên đường baseline
    tx = (w - text_size[0]) // 2 + np.random.randint(-5, 6)
    ty = (h + text_size[1]) // 2
    
    cv2.putText(layer, str(digit), (tx, ty), font, scale, text_color, thickness)
    return layer

def create_synthetic_image(digit, save_path, augmentation_level='train'):
    """
    Tạo ảnh một ô chữ số đồng hồ nước với các hiệu ứng:
    - Số đỏ / Số đen
    - Số đang lăn (Rolling digits)
    - Xoay nghiêng, bóng râm, nhiễu hạt
    """
    h, w = 100, 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX])
    
    # 70% đồng hồ có nền sáng (trắng/xám nhạt), 30% nền tối (đen)
    is_light_bg = random.random() < 0.7 
    
    # 25% tỷ lệ chữ số là màu đỏ (chỉ áp dụng trên nền trắng)
    is_red = is_light_bg and random.random() < 0.25 
    
    # Khởi tạo nền và màu chữ
    if is_light_bg:
        img.fill(random.randint(220, 255)) 
        if is_red:
            # Màu đỏ (BGR)
            text_color = (random.randint(10, 50), random.randint(10, 50), random.randint(180, 255))
        else:
            # Màu đen
            text_color = (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
    else:
        img.fill(random.randint(10, 40))
        text_color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
        
    scale = np.random.uniform(2.2, 2.8)
    thickness = np.random.randint(3, 6)
    
    # --- MÔ PHỎNG SỐ LĂN (ROLLING DIGITS) ---
    # Khoảng 30% ảnh sẽ bị hiệu ứng cuộn số (số trên bị đẩy lên, số dưới chui lên)
    is_rolling = random.random() < 0.3
    
    txt_layer = np.zeros((h, w, 3), dtype=np.uint8) if not is_light_bg else np.full((h, w, 3), 255, dtype=np.uint8)
    
    if is_rolling:
        next_digit = (digit + 1) % 10
        layer1 = draw_digit_on_layer(digit, font, scale, thickness, text_color, is_light_bg, h, w)
        layer2 = draw_digit_on_layer(next_digit, font, scale, thickness, text_color, is_light_bg, h, w)
        
        # Mức độ bị đẩy lên (offset), từ 10px đến 90px
        offset = random.randint(10, h - 10) 
        
        # Ghép nửa trên của số hiện tại
        txt_layer[0:h-offset, :] = layer1[offset:h, :]
        # Ghép nửa dưới của số tiếp theo
        txt_layer[h-offset:h, :] = layer2[0:offset, :]
    else:
        # Nếu không cuộn thì chỉ vẽ số bình thường và có thể trượt nhẹ (shift) y
        txt_layer = draw_digit_on_layer(digit, font, scale, thickness, text_color, is_light_bg, h, w)
        shift_y = random.randint(-10, 10)
        M_shift = np.float32([[1, 0, 0], [0, 1, shift_y]])
        border_val = (255, 255, 255) if is_light_bg else (0, 0, 0)
        txt_layer = cv2.warpAffine(txt_layer, M_shift, (w, h), borderValue=border_val)
    
    # --- XOAY NGHIÊNG ---
    angle = np.random.randint(-8, 9) if augmentation_level == 'train' else np.random.randint(-20, 21)
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
    border_val = (255, 255, 255) if is_light_bg else (0, 0, 0)
    txt_layer = cv2.warpAffine(txt_layer, M, (w, h), borderValue=border_val)
        
    # --- GHÉP CHỮ VÀO NỀN ---
    if is_light_bg:
        # Loại bỏ màu nền trắng của text layer để dán lên img (để giữ lại các nhiễu trên nền img)
        gray_txt = cv2.cvtColor(txt_layer, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(gray_txt, 220, 255, cv2.THRESH_BINARY_INV)[1]
        img_fg = cv2.bitwise_and(txt_layer, txt_layer, mask=mask)
        img_bg = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
        img = cv2.add(img_bg, img_fg)
    else:
        img = cv2.add(img, txt_layer)
        
    # --- HIỆU ỨNG ÁNH SÁNG & BÓNG RÂM ---
    # 1. Bóng râm cố định ở đỉnh và đáy viền nhựa
    shadow_color = (random.randint(150, 190), random.randint(150, 190), random.randint(150, 190)) if is_light_bg else (30, 30, 30)
    cv2.rectangle(img, (0, 0), (w, random.randint(5, 15)), shadow_color, -1)
    cv2.rectangle(img, (0, h-random.randint(5, 15)), (w, h), shadow_color, -1)
    
    # 2. Gradient ánh sáng (đổ bóng nghiêng)
    if random.random() < 0.6:
        shadow = np.zeros((h, w, 3), dtype=np.uint8)
        intensity = random.randint(30, 80)
        for i in range(h):
            factor = i / h
            # Tối dần từ trên xuống dưới
            shadow[i, :] = (int(intensity * factor), int(intensity * factor), int(intensity * factor))
        
        # Nếu random thì lật ngược đổ bóng từ dưới lên
        if random.random() > 0.5:
            shadow = shadow[::-1, :, :]
            
        img = cv2.subtract(img, shadow)

    # --- NHIỄU (NOISE) VÀ MỜ (BLUR) ---
    noise_sigma = random.randint(5, 20)
    noise = np.random.normal(0, noise_sigma, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    blur_k = random.choice([3, 5])
    img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)

    cv2.imwrite(save_path, img)

def generate_datasets():
    base_dir = "datasets"
    
    print(f"Bắt đầu tạo dữ liệu huấn luyện nâng cao tại '{base_dir}'...")
    
    # Số lượng dữ liệu
    TRAIN_SAMPLES_PER_DIGIT = 800  # Total 8000 train images
    TEST_SAMPLES_PER_DIGIT = 100   # Total 1000 test images
    
    # Create structure
    for split in ['train', 'test']:
        for i in range(10):
            path = os.path.join(base_dir, split, str(i))
            ensure_dir(path)

    # Generate Train
    print("Đang tạo tập Training (Có mô phỏng số lăn, số đỏ, bóng râm)...")
    for digit in range(10):
        save_dir = os.path.join(base_dir, 'train', str(digit))
        for i in range(TRAIN_SAMPLES_PER_DIGIT):
            create_synthetic_image(digit, os.path.join(save_dir, f"{digit}_{i}.jpg"), 'train')
            
    # Generate Test
    print("Đang tạo tập Testing (Độ khó cao hơn)...")
    for digit in range(10):
        save_dir = os.path.join(base_dir, 'test', str(digit))
        for i in range(TEST_SAMPLES_PER_DIGIT):
            create_synthetic_image(digit, os.path.join(save_dir, f"{digit}_{i}.jpg"), 'test')

    print("Hoàn tất tạo bộ Dataset mô phỏng thực tế!")

if __name__ == "__main__":
    generate_datasets()
