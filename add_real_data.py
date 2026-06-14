import cv2
import os
import shutil
from segmentation import segment_meter_digits

def add_real_image_to_train(image_path, digits_sequence):
    """
    Cắt ảnh thật và đưa vào thư mục huấn luyện tương ứng.
    digits_sequence: Chuỗi các số thật có trong ảnh (VD: "66667")
    """
    print(f"Bo sung du lieu that tu anh {image_path}...")
    if not os.path.exists(image_path):
        print(f"Khong tim thay file: {image_path}")
        return
        
    try:
        digits_images = segment_meter_digits(image_path, num_digits=len(digits_sequence), margin_ratio=0.22)
    except Exception as e:
        print(f"Loi cat anh: {e}")
        return
        
    for i, (digit_char, digit_img) in enumerate(zip(digits_sequence, digits_images)):
        save_dir = os.path.join("datasets", "train", digit_char)
        os.makedirs(save_dir, exist_ok=True)
        
        # Để đảm bảo AI học kỹ chữ này, chúng ta sẽ nhân bản nó ra làm 50 tấm với độ sáng/nhiễu khác nhau (Data Augmentation)
        print(f"Dang sinh them 50 bien the cho so {digit_char}...")
        for j in range(50):
            # Tạo biến thể
            img_aug = digit_img.copy()
            
            # Thêm nhiễu ngẫu nhiên
            noise = np.random.normal(0, np.random.randint(1, 10), img_aug.shape).astype(np.uint8)
            img_aug = cv2.add(img_aug, noise)
            
            # Lưu file
            save_path = os.path.join(save_dir, f"real_data_{i}_{j}.jpg")
            cv2.imwrite(save_path, img_aug)
            
    print("Hoan tat them du lieu that!")

if __name__ == "__main__":
    import numpy as np
    
    # Giả định file debug_images/0_anh_goc_tu_java.jpg là tấm ảnh 66667 mà user gửi
    # Nếu user lưu với tên khác, vui lòng đổi tên ở đây
    image_name = os.path.join("debug_images", "0_anh_goc_tu_java.jpg")
    sequence = "66667"
    
    add_real_image_to_train(image_name, sequence)
