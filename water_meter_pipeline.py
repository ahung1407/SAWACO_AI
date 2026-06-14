
import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. Preprocessing
# ==========================================
def preprocess_for_gen(img):
    """
    Preprocessing function for ImageDataGenerator.
    Input: Image array (H, W, C) - already grayscale (H, W, 1) from flow_from_directory
    Output: (H, W, 1) float32 in [0, 1].
    """
    # img comes in as float32 if flow_from_directory used default? 
    # Actually IDG passes float if rescale is set, or uint8. 
    # We didn't set rescale. So it is uint8.
    
    # Ensure usage of numpy
    img = np.array(img, dtype=np.uint8)

    # [NEW] Chuyển ảnh màu sang xám bằng np.min để số màu đỏ đen lại (giống logic inference)
    if img.ndim == 3 and img.shape[-1] == 3:
        gray = np.min(img, axis=-1).astype(np.uint8)
    else:
        gray = img.squeeze()

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Normalize
    enhanced = enhanced.astype('float32') / 255.0
    enhanced = np.expand_dims(enhanced, axis=-1)
    
    return enhanced

def preprocess_image(image_path):
    """
    Reads an image from path and applies full preprocessing for inference.
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image not found: {image_path}")
        
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Convert to Grayscale using np.min (same as inference_lib)
    if len(img.shape) == 3:
        gray = np.min(img, axis=2).astype(np.uint8)
    else:
        gray = img
    
    # Resize to 28x28
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)

    # Normalize
    processed = enhanced.astype('float32') / 255.0
    processed = np.expand_dims(processed, axis=-1)
    
    return processed

# ==========================================
# 2. Model Architecture (LeNet-5)
# ==========================================
def build_modern_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        
        # Fully Connected
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5), # Chống học vẹt (overfitting)
        
        layers.Dense(10, activation='softmax')
    ])
    return model

# ==========================================
# 3. Data Loading (From Disk)
# ==========================================
def get_data_generators(train_dir='datasets/train', test_dir='datasets/test', batch_size=32):
    """
    Creates generators reading directly from disk folders.
    structure: datasets/train/0/xxx.jpg
    """
    
    # Augmentation for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_for_gen,
        rotation_range=15,
        zoom_range=0.1,
        shear_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    # Only preprocessing for testing
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_for_gen
    )

    print(f"Loading Training Data from {train_dir}...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(28, 28),
        color_mode='rgb', # Load RGB so preprocess can use np.min
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=True
    )

    print(f"Loading Test Data from {test_dir}...")
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(28, 28),
        color_mode='rgb', # Load RGB so preprocess can use np.min
        batch_size=batch_size,
        class_mode='sparse',
        shuffle=False
    )

    class OneChannelWrapper(tf.keras.utils.Sequence):
        def __init__(self, generator):
            self.generator = generator
        def __len__(self):
            return len(self.generator)
        def __getitem__(self, idx):
            x, y = self.generator[idx]
            # Giữ lại 1 channel (vì hàm preprocess trả về 1 nhưng Keras broadcast thành 3 do color_mode='rgb')
            return x[:, :, :, :1], y
        def on_epoch_end(self):
            self.generator.on_epoch_end()

    return OneChannelWrapper(train_generator), OneChannelWrapper(test_generator)

# ==========================================
# 4. Business Logic (Smart Read)
# ==========================================
def smart_read(prev_digit_probs, confidence_gap_threshold=0.2):
    top_indices = np.argsort(prev_digit_probs)[-2:] 
    top1_digit = top_indices[1]
    top2_digit = top_indices[0]
    prob1 = prev_digit_probs[top1_digit]
    prob2 = prev_digit_probs[top2_digit]
    
    confidence_gap = prob1 - prob2
    
    if confidence_gap < 0.5:
        print(f"  [Debug] Top1={top1_digit}({prob1:.2f}), Top2={top2_digit}({prob2:.2f}), Gap={confidence_gap:.2f}")

    if confidence_gap < confidence_gap_threshold:
        is_sequential = False
        if abs(top1_digit - top2_digit) == 1: is_sequential = True
        if {top1_digit, top2_digit} == {0, 9}: is_sequential = True
            
        if is_sequential:
            print("  -> Rolling Digit Detected! Applying Floor/Wrap Logic.")
            if {top1_digit, top2_digit} == {0, 9}: return 9 
            return min(top1_digit, top2_digit)
            
    return top1_digit

# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    
    # Mode 1: Inference on Image
    if len(sys.argv) > 1 and sys.argv[1].endswith(('.jpg', '.png', '.jpeg')):
        image_path = sys.argv[1]
        model_path = 'water_meter_modern.keras'
        
        if not os.path.exists(model_path):
            print("Model not found. Please run without arguments first to train.")
            sys.exit(1)
            
        print(f"Processing: {image_path}")
        model = models.load_model(model_path)
        
        try:
            processed = preprocess_image(image_path)
            input_tensor = np.expand_dims(processed, axis=0)
            probs = model.predict(input_tensor)[0]
            result = smart_read(probs)
            print(f">>> RESULT: {result}")
        except Exception as e:
            print(f"Error: {e}")

    # Mode 2: Train
    else:
        print("=== Training Mode ===")
        if not os.path.exists('datasets/train'):
            print("Error: 'datasets/train' not found. Run 'generate_datasets.py' first.")
            sys.exit(1)
            
        model = build_modern_cnn_model()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        train_gen, test_gen = get_data_generators()
        
        print("\nTraining for 15 epochs...")
        model.fit(train_gen, epochs=15, validation_data=test_gen)
        
        model.save('water_meter_modern.keras')
        print("\nModel saved to 'water_meter_modern.keras'")
        
        # Eval
        loss, acc = model.evaluate(test_gen)
        print(f"Final Test Accuracy: {acc*100:.2f}%")
