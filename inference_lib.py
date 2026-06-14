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

        # Chuyển sang ảnh xám bằng cách lấy giá trị nhỏ nhất của 3 kênh màu (R, G, B)
        # Cách này giúp CẢ chữ số màu đen và chữ số màu đỏ đều trở nên rất đậm (màu đen) trên nền trắng
        if len(img.shape) == 3:
            gray = np.min(img, axis=2).astype(np.uint8)
        else:
            gray = img
            
        # [QUAN TRỌNG] Đã xoá logic tự động đảo màu ảnh (auto-invert)
        # Vì hiện tại mô hình đã được huấn luyện tốt với cả ảnh nền trắng chữ đen và nền đen chữ trắng
        # Việc ép đảo màu sẽ làm sai lệch dữ liệu thật.
        
        # Resize to 28x28
        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

        # CLAHE (Contrast Enhancement)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)

        # Normalize to [0, 1]
        processed = enhanced.astype('float32') / 255.0
        
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
        
        # Debug/Log could go here
        
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
        
        # Predict
        probs = self.model.predict(tensor, verbose=0)[0]
        
        # Apply Logic
        final_digit = self.smart_read_logic(probs)
        
        return {
            "digit": int(final_digit),
            "confidence": float(np.max(probs)),
            "all_probs": [float(p) for p in probs]
        }
