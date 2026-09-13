import os
# Tắt bớt log rác của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import logging
logging.getLogger("absl").setLevel(logging.ERROR)

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class WaterMeterReader:
    def __init__(self, model_path='water_meter_modern.keras'):
        """
        Initializes the Water Meter Reader.
        Loads the model once to be reused for multiple predictions.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}. Please train the model first.")
        
        print(f"Loading Water Meter Model from {model_path}...")
        self.model = load_model(model_path)
        print("Model loaded successfully.")

    def preprocess(self, image_input):
        """
        Preprocesses an image for the model.
        Args:
            image_input: Can be a file path (str) or a numpy array (image).
        Returns:
            Preprocessed tensor (1, 28, 28, 1) ready for prediction.
        """
        # Load image if path provided
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise ValueError(f"Image not found: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError("Could not read image file.")
        else:
            # Assume it's a numpy array
            img = image_input

        # Chuyển sang ảnh xám
        if len(img.shape) == 3:
            # Dùng np.min để số đỏ cũng ra đen (phù hợp với dataset training)
            gray = np.min(img, axis=2).astype(np.uint8)
        else:
            gray = img

        # ----------------------------------------------------------------
        # Chuẩn hoá độ tương phản cục bộ bằng CLAHE
        # (Digit đã được extract_clean_digit_ink làm sạch viền nhựa trước đó)
        # ----------------------------------------------------------------

        # Chuẩn hoá độ tương phản cục bộ bằng CLAHE (giam clip xuong 2.0 de it nhieu hon)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        # Normalize
        processed = resized.astype('float32') / 255.0
        
        # Reshape to (1, 28, 28, 1) for batch prediction
        input_tensor = np.expand_dims(processed, axis=-1)
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor

    def smart_read_logic(self, probs, confidence_threshold=0.2):
        """
        Applies business logic for rolling digits.
        """
        top_indices = np.argsort(probs)[-2:] 
        top1 = top_indices[1]
        top2 = top_indices[0]
        prob1 = probs[top1]
        prob2 = probs[top2]
            
        confidence_gap = prob1 - prob2
        
        if confidence_gap < confidence_threshold:
            # Check sequentiality
            is_seq = False
            if abs(top1 - top2) == 1: is_seq = True
            if {top1, top2} == {0, 9}: is_seq = True
            
            if is_seq:
                # Apply Rolling Logic
                if {top1, top2} == {0, 9}: return 9
                return min(top1, top2)
        
        return top1

    def predict(self, image_input):
        """
        End-to-end prediction: Preprocess -> Model -> Logic
        """
        tensor = self.preprocess(image_input)
        
        # [NEW] Nếu ảnh truyền vào đen hoàn toàn (bị Geometry Gate từ chối)
        if np.max(tensor) == 0.0:
            return {
                "digit": 'NaN',
                "confidence": 1.0,
                "all_probs": [0.0]*10
            }
        
        # Predict
        probs = self.model.predict(tensor, verbose=0)[0]
        
        # Apply Logic
        final_digit = self.smart_read_logic(probs)
        
        return {
            "digit": final_digit,
            "confidence": float(np.max(probs)),
            "all_probs": [float(p) for p in probs]
        }
        
    def predict_sequence(self, digit_images):
        """
        Dự đoán toàn bộ 5 chữ số và áp dụng Luật Bánh Răng Cơ Học
        """
        results = []
        for img in digit_images:
            res = self.predict(img)
            # Nếu có bất kỳ ô nào bị lá cây che khuất, dừng toàn bộ và báo lỗi
            if res['digit'] == 'NaN':
                return None
            results.append(res)
            
        # ÁP DỤNG LUẬT BÁNH RĂNG (Mechanical Gear Logic)
        # Duyệt từ trái sang phải (trừ số thập phân cuối cùng)
        for i in range(len(results) - 1):
            current_res = results[i]
            next_res = results[i+1]
            
            # Nếu chữ số hiện tại đang phân vân (confidence thấp)
            if current_res['confidence'] < 0.6:
                # Kiểm tra chữ số bên phải nó có đang ở thời điểm "Roll" không (số 9 hoặc 0)
                next_digit = next_res['digit']
                if next_digit not in [9, 0]:
                    # Chữ số bên phải không lăn -> Sự phân vân của chữ số hiện tại là do NHIỄU, không phải số lăn.
                    # Khôi phục chữ số hiện tại về giá trị chắc chắn nhất.
                    probs = np.array(current_res['all_probs'])
                    current_res['digit'] = int(np.argmax(probs))
                    current_res['confidence'] = 0.99
                    
        return results

    def read_billing_meter(self, image_input):
        """
        Nghiệp vụ đọc đồng hồ nước chuẩn Sawaco:
        - Phân biệt rõ ràng:
          + 4 ô màu đen đầu tiên: Chỉ số khối nước m3 chính thức dùng để tính hóa đơn tiền nước.
          + Ô màu đỏ thứ 5: Chỉ số phụ (hàng 0.1 m3 = 100 lít), mang tính tham khảo kỹ thuật.
        - Áp dụng nguyên tắc Làm tròn sàn (Floor Rule) bảo vệ người tiêu dùng khi bánh xe đen đang ở pha nhảy số dở dang.
        """
        if isinstance(image_input, (str, np.ndarray)) and not isinstance(image_input, list):
            from segmentation import segment_meter_digits
            digit_images = segment_meter_digits(image_input, num_digits=5, margin_ratio=0.22)
        else:
            digit_images = image_input

        seq_results = self.predict_sequence(digit_images)
        if seq_results is None:
            return None

        # 4 ô màu đen đầu tiên (indices 0, 1, 2, 3) là chỉ số m3
        black_digits = [r['digit'] for r in seq_results[:4]]
        red_digit = seq_results[4]['digit'] if len(seq_results) >= 5 else 0

        # Ràng buộc cơ học làm tròn sàn (Floor Rule) bảo vệ người tiêu dùng:
        # Chuỗi truyền động cơ học từ phải qua trái:
        # - Ô 4 (Số đỏ) -> Ô 3 (Đen hàng đơn vị 1 m3)
        # - Ô 3 (Đen đơn vị) -> Ô 2 (Đen hàng chục 10 m3)
        # - Ô 2 (Đen hàng chục) -> Ô 1 (Đen hàng trăm 100 m3)
        # - Ô 1 (Đen hàng trăm) -> Ô 0 (Đen hàng ngàn 1000 m3)
        #
        # Nguyên tắc: Nếu bánh xe bên phải đang ở số 9 (chưa hoàn tất bước nhảy qua 0),
        # bánh xe bên trái kế bên nếu có dấu hiệu nhấp nhô phân vân (confidence < 0.75)
        # BẮT BUỘC phải làm tròn xuống số nhỏ hơn (số cũ)!
        
        # 1. Ràng buộc từ số ĐỎ (ô 4) sang số ĐEN hàng đơn vị (ô 3):
        if len(seq_results) >= 5:
            red_res = seq_results[4]
            unit_res = seq_results[3]
            if red_res['digit'] == 9 and unit_res['confidence'] < 0.75:
                probs = unit_res['all_probs']
                top2 = np.argsort(probs)[-2:]
                black_digits[3] = int(min(top2[0], top2[1]))

        # 2. Ràng buộc liên tầng giữa các ô màu ĐEN (từ ô 3 về ô 0):
        for i in range(2, -1, -1):
            right_digit = black_digits[i+1]
            left_res = seq_results[i]
            if right_digit == 9 and left_res['confidence'] < 0.75:
                probs = left_res['all_probs']
                top2 = np.argsort(probs)[-2:]
                black_digits[i] = int(min(top2[0], top2[1]))

        billing_m3_str = "".join(str(d) for d in black_digits)
        billing_m3_val = int(billing_m3_str) if billing_m3_str.isdigit() else 0
        red_digit_str = str(red_digit)
        full_reading = f"{billing_m3_str}{red_digit_str}"
        formatted = f"{billing_m3_str}.{red_digit_str} m3"
        avg_conf = float(np.mean([r['confidence'] for r in seq_results]))

        return {
            "billing_m3": billing_m3_val,
            "billing_m3_str": billing_m3_str,
            "fraction_liters": int(red_digit_str) * 100 if red_digit_str.isdigit() else 0,
            "fraction_digit": red_digit,
            "full_reading": full_reading,
            "formatted": formatted,
            "confidence": avg_conf,
            "digits": seq_results
        }
