"""
retrain_with_real_data.py
--------------------------
1. Load synthetic dataset (data/datasets/)
2. Load real labeled digits (data/real_digits_labeled/) 
3. Augment real images x200 each (heavy augmentation)
4. Mix together và retrain model
5. Save model mới
"""
import os, sys
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

REAL_DATA_DIR  = "data/real_digits_labeled"
SYNTH_DATA_DIR = "data/datasets/train"
MODEL_PATH     = "water_meter_modern.keras"
OUTPUT_MODEL   = "water_meter_modern.keras"
IMG_SIZE       = 28
AUGMENT_TIMES  = 200
EPOCHS         = 30
BATCH_SIZE     = 64

def augment_image(img, seed=None):
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    angle = rng.uniform(-12, 12)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    aug = cv2.warpAffine(img, M, (w, h), borderValue=255)
    scale = rng.uniform(0.85, 1.15)
    new_w, new_h = int(w*scale), int(h*scale)
    if new_w > 0 and new_h > 0:
        scaled = cv2.resize(aug, (new_w, new_h))
        if scale >= 1.0:
            x1 = (new_w - w)//2
            y1 = (new_h - h)//2
            aug = scaled[y1:y1+h, x1:x1+w]
        else:
            pad_x = (w - new_w)//2
            pad_y = (h - new_h)//2
            aug = cv2.copyMakeBorder(scaled, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x,
                                     cv2.BORDER_CONSTANT, value=255)
    tx = rng.uniform(-w*0.05, w*0.05)
    ty = rng.uniform(-h*0.05, h*0.05)
    aug = cv2.warpAffine(aug, np.float32([[1,0,tx],[0,1,ty]]), (w, h), borderValue=255)
    if rng.random() < 0.5:
        aug = cv2.GaussianBlur(aug, (rng.choice([3,5]),)*2, 0)
    aug = np.clip(aug.astype(float)*rng.uniform(0.80,1.20), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        noise = rng.normal(0, 8, aug.shape).astype(np.int16)
        aug = np.clip(aug.astype(np.int16)+noise, 0, 255).astype(np.uint8)
        
    # Mo phong vet loa sang den Flash ngang qua tam chu so (35% xac suat)
    if rng.random() < 0.35:
        glare_y = int(rng.uniform(0.35, 0.65) * h)
        glare_th = max(2, int(rng.uniform(3, 8)))
        aug[glare_y:glare_y+glare_th, :] = 255
        
    # Mo phong vet vien nhua con sot o mep (20% xac suat)
    if rng.random() < 0.20:
        line_w = int(rng.uniform(2, 5))
        if rng.random() < 0.5:
            aug[:, :line_w] = int(rng.uniform(30, 80))
        else:
            aug[:, w-line_w:] = int(rng.uniform(30, 80))
            
    return aug

def preprocess_for_model(img_bgr):
    gray = np.min(img_bgr, axis=2).astype(np.uint8) if len(img_bgr.shape)==3 else img_bgr
    _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    dk = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    bright_mask = cv2.dilate(bright_mask, dk, iterations=3)
    gray[bright_mask == 0] = 255
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    gray = clahe.apply(gray)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype('float32') / 255.0

print("="*50)
print("BUOC 1: Load synthetic dataset...")
X_synth, y_synth = [], []
if os.path.exists(SYNTH_DATA_DIR):
    for label in range(10):
        d = os.path.join(SYNTH_DATA_DIR, str(label))
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            if not f.lower().endswith(('.jpg','.png')): continue
            img = cv2.imread(os.path.join(d, f))
            if img is None: continue
            X_synth.append(preprocess_for_model(img))
            y_synth.append(label)
    print(f"  Synthetic: {len(X_synth)} anh")

print("\nBUOC 2: Load va augment real data (Can bang 600 anh moi chu so)...")
X_real, y_real = [], []
TARGET_PER_CLASS = 600

if os.path.exists(REAL_DATA_DIR):
    for label in range(10):
        d = os.path.join(REAL_DATA_DIR, str(label))
        if not os.path.exists(d): continue
        files = [f for f in os.listdir(d) if f.lower().endswith(('.jpg','.png'))]
        if not files: continue
        
        loaded_imgs = []
        for f in files:
            img = cv2.imread(os.path.join(d, f))
            if img is not None:
                loaded_imgs.append(img)
                X_real.append(preprocess_for_model(img))
                y_real.append(label)
                
        if not loaded_imgs:
            continue
            
        needed = TARGET_PER_CLASS - len(loaded_imgs)
        for k in range(max(0, needed)):
            src_img = loaded_imgs[k % len(loaded_imgs)]
            aug = augment_image(src_img, seed=k * 137 + label * 31)
            X_real.append(preprocess_for_model(aug))
            y_real.append(label)
            
        print(f"  Digit {label}: {len(loaded_imgs)} mau goc -> {TARGET_PER_CLASS} mau da can bang")

print(f"\nBUOC 3: Ket hop dataset...")
X_all = np.array(X_synth + X_real)[..., np.newaxis]
y_all = np.array(y_synth + y_real)
print(f"  Tong: {len(X_all)} anh | Phan phoi: {np.bincount(y_all)}")

idx = np.random.permutation(len(X_all))
X_all, y_all = X_all[idx], y_all[idx]
split = int(len(X_all)*0.9)
X_tr, X_v = X_all[:split], X_all[split:]
y_tr, y_v = y_all[:split], y_all[split:]

print(f"\nBUOC 4: Load model va fine-tune (lr=1e-4)...")
model = load_model(MODEL_PATH)
# [FIX] Keras 3 compatible: recompile voi learning rate moi
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_tr, to_categorical(y_tr, 10),
    validation_data=(X_v, to_categorical(y_v, 10)),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ],
    verbose=1
)

print(f"\nBest val_acc: {max(history.history['val_accuracy'])*100:.1f}%")
model.save(OUTPUT_MODEL)
print(f"Da luu: {OUTPUT_MODEL}")
print("HOAN THANH! Restart api_server.py de dung model moi.")
