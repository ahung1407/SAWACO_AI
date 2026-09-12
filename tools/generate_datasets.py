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
        
    # [NEW] Giữ lại dải scale rộng và thickness từ rất mỏng (2) đến rất dày (12)
    # để AI có thể đọc được CẢ font mỏng (như image.png) VÀ font béo (như test_1.jpg)
    scale = np.random.uniform(2.2, 3.8)
    thickness = np.random.randint(2, 12)
    
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

def create_nan_image(save_path):
    """
    Tạo ảnh lớp NaN mô phỏng bị che khuất bởi bùn đất, lá cây, chói lóa,
    hoặc chỉ là nhiễu hạt không có số rõ ràng.
    """
    h, w = 100, 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    type_nan = random.choice(['noise', 'mud', 'leaf', 'glare', 'scratches'])
    
    if type_nan == 'noise':
        img.fill(random.randint(50, 200))
        noise = np.random.normal(0, random.randint(30, 80), img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    elif type_nan == 'mud':
        # Bùn đất dính
        img.fill(random.randint(180, 255))
        mud_color = (random.randint(10, 50), random.randint(30, 70), random.randint(40, 90)) # Nâu đất
        for _ in range(random.randint(3, 8)):
            cx, cy = random.randint(0, w), random.randint(0, h)
            rx, ry = random.randint(10, 40), random.randint(10, 40)
            cv2.ellipse(img, (cx, cy), (rx, ry), random.randint(0, 180), 0, 360, mud_color, -1)
    elif type_nan == 'leaf':
        # Lá cây xanh
        img.fill(random.randint(200, 255))
        leaf_color = (random.randint(0, 50), random.randint(100, 200), random.randint(0, 50))
        for _ in range(random.randint(1, 3)):
            pts = np.array([
                [random.randint(-20, w), random.randint(-20, h)],
                [random.randint(-20, w+20), random.randint(-20, h+20)],
                [random.randint(-20, w), random.randint(h//2, h+20)]
            ], np.int32)
            cv2.fillPoly(img, [pts], leaf_color)
    elif type_nan == 'glare':
        # Chói lóa trắng xóa
        img.fill(random.randint(0, 50))
        cv2.circle(img, (w//2, h//2), random.randint(20, 60), (255, 255, 255), -1)
        img = cv2.GaussianBlur(img, (31, 31), 0)
    elif type_nan == 'scratches':
        img.fill(random.randint(200, 255))
        for _ in range(random.randint(10, 30)):
            pt1 = (random.randint(0, w), random.randint(0, h))
            pt2 = (random.randint(0, w), random.randint(0, h))
            color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            cv2.line(img, pt1, pt2, color, random.randint(1, 4))
            
    # Thêm chút nhiễu chung
    blur_k = random.choice([3, 5, 7])
    img = cv2.GaussianBlur(img, (blur_k, blur_k), 0)
    
    cv2.imwrite(save_path, img)

def generate_datasets():
    base_dir = "datasets"
    
    print(f"Bat dau tao du lieu huan luyen nang cao tai '{base_dir}'...")
    
    # Số lượng dữ liệu
    TRAIN_SAMPLES_PER_DIGIT = 800  # Total 8000 train images
    TEST_SAMPLES_PER_DIGIT = 100   # Total 1000 test images
    
    # Create structure
    for split in ['train', 'test']:
        for i in range(10):
            path = os.path.join(base_dir, split, str(i))
            ensure_dir(path)
        ensure_dir(os.path.join(base_dir, split, 'NaN'))

    # Generate Train
    print("Dang tao tap Training (Co mo phong so lan, so do, bong ram)...")
    for digit in range(10):
        save_dir = os.path.join(base_dir, 'train', str(digit))
        for i in range(TRAIN_SAMPLES_PER_DIGIT):
            create_synthetic_image(digit, os.path.join(save_dir, f"{digit}_{i}.jpg"), 'train')
            
    # Generate NaN (Train)
    nan_train_dir = os.path.join(base_dir, 'train', 'NaN')
    for i in range(TRAIN_SAMPLES_PER_DIGIT):
        create_nan_image(os.path.join(nan_train_dir, f"nan_{i}.jpg"))
            
    # Generate Test
    print("Dang tao tap Testing (Do kho cao hon)...")
    for digit in range(10):
        save_dir = os.path.join(base_dir, 'test', str(digit))
        for i in range(TEST_SAMPLES_PER_DIGIT):
            create_synthetic_image(digit, os.path.join(save_dir, f"{digit}_{i}.jpg"), 'test')
            
    # Generate NaN (Test)
    nan_test_dir = os.path.join(base_dir, 'test', 'NaN')
    for i in range(TEST_SAMPLES_PER_DIGIT):
        create_nan_image(os.path.join(nan_test_dir, f"nan_{i}.jpg"))

    print("Hoan tat tao bo Dataset mo phong thuc te!")

if __name__ == "__main__":
    generate_datasets()
