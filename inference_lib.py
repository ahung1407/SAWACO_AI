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
        # [FIX] Loai bo vien nhua truoc CLAHE:
        # Tim vung cua so sang (> 160) va set tat ca vung toi ben ngoai thanh trang
        # Ngan chan vien nhua tao ra net thua khien CNN nham 0->6, 0->8
        # ----------------------------------------------------------------
        _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        # Dilate de lap day khoang trong trong vung sang
        dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        bright_mask = cv2.dilate(bright_mask, dilate_k, iterations=3)
        # Vung toi (vien nhua) -> trang (nền)
        gray[bright_mask == 0] = 255

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
