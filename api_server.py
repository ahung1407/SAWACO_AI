import sys
import os
import shutil
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager

try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', line_buffering=True)
except Exception:
    pass

import numpy as np
import cv2
import uvicorn
from fastapi import FastAPI, UploadFile, File

from segmentation import segment_meter_digits
from inference_lib import WaterMeterReader
from tunnel_manager import start_tunnel_manager, stop_tunnel_manager

# Filter out periodic /health access logs to keep console clean
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/health" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())

def save_debug_session(img, digits_images, sequence_results, final_number_str, ai_detected_number, original_filename=""):
    try:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
        base_dir = "debug_images"
        session_dir = os.path.join(base_dir, timestamp_str)
        latest_dir = os.path.join(base_dir, "latest")
        
        os.makedirs(session_dir, exist_ok=True)
        os.makedirs(latest_dir, exist_ok=True)
        
        # 1. Original image
        cv2.imwrite(os.path.join(session_dir, "0_original.jpg"), img)
        cv2.imwrite(os.path.join(latest_dir, "0_original.jpg"), img)
        cv2.imwrite(os.path.join(base_dir, "0_original.jpg"), img)
        
        # 2. Box segmentation visual
        if os.path.exists("debug_final_boxes.jpg"):
            shutil.copy("debug_final_boxes.jpg", os.path.join(session_dir, "0_boxes.jpg"))
            shutil.copy("debug_final_boxes.jpg", os.path.join(latest_dir, "0_boxes.jpg"))
            
        # 3. Individual cropped digit boxes
        if digits_images:
            for i, digit_img in enumerate(digits_images):
                fname = f"1_digit_{i+1}.jpg"
                cv2.imwrite(os.path.join(session_dir, fname), digit_img)
                cv2.imwrite(os.path.join(latest_dir, fname), digit_img)
                cv2.imwrite(os.path.join(base_dir, fname), digit_img)
                
        # 4. JSON summary
        serializable_details = []
        if sequence_results:
            for r in sequence_results:
                serializable_details.append({
                    "digit": int(r['digit']) if str(r['digit']).isdigit() else str(r['digit']),
                    "confidence": float(r.get('confidence', 0.0)),
                    "all_probs": [round(float(p), 4) for p in r.get('all_probs', [])]
                })

        meta = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_folder": timestamp_str,
            "filename": original_filename,
            "raw_string": final_number_str,
            "water_reading": ai_detected_number,
            "details": serializable_details
        }
        with open(os.path.join(session_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        with open(os.path.join(latest_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
            
        # 5. Housekeeping: retain latest 50 sessions
        all_dirs = sorted([
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d != "latest"
        ])
        if len(all_dirs) > 50:
            for old_dir in all_dirs[:-50]:
                shutil.rmtree(os.path.join(base_dir, old_dir), ignore_errors=True)
                
        print(f"[DEBUG] Session saved to: {session_dir}", flush=True)
    except Exception as e:
        print(f"[ERROR] Failed to save debug session: {e}", flush=True)

# ==========================================
# 1. INIT AI MODEL
# ==========================================
model_path = 'water_meter_modern.keras'
if os.path.exists(model_path):
    meter_reader = WaterMeterReader(model_path=model_path)
    print("AI Model loaded successfully.", flush=True)
else:
    print(f"[ERROR] Model file not found at {model_path}. Please train the model first.", flush=True)
    meter_reader = None

# ==========================================
# 2. FASTAPI LIFESPAN & ROUTES
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start auto-recovering tunnel
    start_tunnel_manager(port=8000, subdomain="ai-sawaco")
    yield
    # Clean shutdown of tunnel processes
    stop_tunnel_manager()

app = FastAPI(title="Water Meter AI OCR Service", lifespan=lifespan)

@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "SAWACO Water Meter AI OCR",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/api/ai/ocr")
async def process_ocr(file: UploadFile = File(...)):
    try:
        print(f"[INFO] Received image from backend: {file.filename}", flush=True)
        
        # 1. Read raw bytes
        image_bytes = await file.read()
        
        # 2. Decode image directly in memory
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "Uploaded file is not a valid image"}

        print(f"[INFO] Image decoded successfully. Resolution: {img.shape}", flush=True)

        if meter_reader is None:
            return {"status": "error", "message": "AI Model is not loaded"}
        
        # 3. Segment into 5 digit boxes
        try:
            digits_images = segment_meter_digits(img, num_digits=5, margin_ratio=0.22)
        except Exception as e:
            return {"status": "error", "message": f"Image segmentation failed: {str(e)}"}
            
        # 4. Predict digits with Sawaco billing rules and mechanical floor rule
        billing_result = meter_reader.read_billing_meter(digits_images)
        
        # Check if digits are obscured (leaf, mud, reflection)
        if billing_result is None:
            print("[WARNING] Character obscured or contaminated (NaN detected).", flush=True)
            save_debug_session(img, digits_images, None, "NaN", 0.0, file.filename)
            return {
                "status": "warning",
                "message": "Digits obscured by dirt or foreign objects. Please clean and recapture.",
                "water_reading": 0.0,
                "billing_m3": 0
            }
            
        final_number_str = billing_result["full_reading"]
        ai_detected_number = float(f"{billing_result['billing_m3_str']}.{billing_result['fraction_digit']}")
        billing_m3 = billing_result["billing_m3"]
        sequence_results = billing_result["digits"]
            
        print(f"[INFO] Raw sequence: {final_number_str} | Billing m3: {billing_m3} | Display: {billing_result['formatted']}", flush=True)

        # 5. Save debug artifacts
        save_debug_session(img, digits_images, sequence_results, final_number_str, ai_detected_number, file.filename)
        
        # 6. Return JSON response (backward compatible + new Sawaco billing standard)
        return {
            "status": "success",
            "water_reading": ai_detected_number,
            "billing_m3": billing_m3,
            "billing_m3_str": billing_result["billing_m3_str"],
            "fraction_liters": billing_result["fraction_liters"],
            "formatted": billing_result["formatted"],
            "message": "Recognition successful",
            "raw_string": final_number_str
        }
        
    except Exception as e:
        print(f"[ERROR] Inference exception: {str(e)}", flush=True)
        return {
            "status": "error",
            "message": str(e),
            "water_reading": 0.0
        }

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)
