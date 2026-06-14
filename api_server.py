from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
import os

from segmentation import segment_meter_digits
from inference_lib import WaterMeterReader

# ======================================================================
# KHỞI TẠO MODEL AI Ở ĐÂY (Để model chỉ load 1 lần vào RAM khi bật Server)
# ==========================================
# 1. INIT AI MODEL
# ==========================================
model_path = 'water_meter_modern.keras'
if os.path.exists(model_path):
    meter_reader = WaterMeterReader(model_path=model_path)
    print("Khởi tạo AI Model thành công!")
else:
    print(f"LỖI: Không tìm thấy model tại {model_path}. Hãy train trước!")
    meter_reader = None

app = FastAPI(title="Water Meter AI OCR Service")

@app.post("/api/ai/ocr")
async def process_ocr(file: UploadFile = File(...)):
    try:
        print(f"📥 BẮT ĐẦU NHẬN ẢNH TỪ BACKEND: {file.filename}")
        
        # 1. Đọc dữ liệu byte thô từ Spring Boot ném sang
        image_bytes = await file.read()
        
        # CHUYỂN ĐỔI ẢNH CHO AI ĐỌC TRỰC TRỰC TIẾP TRÊN RAM
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"status": "error", "message": "File gửi sang không đọc được dạng ảnh"}

        print(f"✅ Đã giải mã ảnh thành công. Kích thước: {img.shape}")

        # =================================================================
        # [DEBUG] LƯU ẢNH ĐỂ QUAN SÁT BẰNG MẮT THƯỜNG
        # =================================================================
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        # Lưu ảnh gốc gửi từ Java sang
        cv2.imwrite(os.path.join(debug_dir, "0_anh_goc_tu_java.jpg"), img)

        if meter_reader is None:
            return {"status": "error", "message": "AI Model chưa được load!"}

        # =================================================================
        # 2. KHU VỰC CỦA NGƯỜI LÀM AI: CHÈN THUẬT TOÁN NHẬN DIỆN VÀO ĐÂY
        # =================================================================
        
        # Bước 2.1: Cắt ảnh thành 5 chữ số
        try:
            digits_images = segment_meter_digits(img, num_digits=5, margin_ratio=0.22)
        except Exception as e:
            return {"status": "error", "message": f"Lỗi lúc cắt ảnh: {str(e)}"}
            
        # Bước 2.2: Đọc từng chữ số
        final_number_str = ""
        for i, digit_img in enumerate(digits_images):
            # [DEBUG] Lưu từng ảnh chữ số đã bị cắt ra
            cv2.imwrite(os.path.join(debug_dir, f"1_chu_so_thu_{i+1}.jpg"), digit_img)
            
            result = meter_reader.predict(digit_img)
            final_number_str += str(result['digit'])
            
        print(f"Chuỗi số thô đọc được: {final_number_str}")
        
        # Bước 2.3: Chuyển đổi thành số m3. 
        # Đồng hồ thường có 4 số đen (khối) và 1 số đỏ (trăm lít = 0.1 m3)
        # Ví dụ chuỗi "05580" -> 0558.0 m3
        try:
            # Lấy 4 số đầu làm phần nguyên, số cuối làm phần thập phân
            if len(final_number_str) == 5:
                integer_part = final_number_str[:4]
                decimal_part = final_number_str[4:]
                ai_detected_number = float(f"{integer_part}.{decimal_part}")
            else:
                ai_detected_number = float(final_number_str)
        except ValueError:
            ai_detected_number = 0.0
            
        print(f"🎯 KẾT QUẢ AI ĐỌC ĐƯỢC: {ai_detected_number} m3")
        
        # 3. TRẢ KẾT QUẢ VỀ CHO JAVA BACKEND
        return {
            "status": "success",
            "water_reading": ai_detected_number,
            "message": "Đã nhận diện thành công",
            "raw_string": final_number_str # Gửi kèm chuỗi thô để dễ debug
        }
        
    except Exception as e:
        print(f"❌ LỖI TRONG QUÁ TRÌNH XỬ LÝ AI: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "water_reading": 0.0
        }

if __name__ == "__main__":
    # Lắng nghe ở cổng 8000, khớp với http://localhost:8000/api/ai/ocr bên Java
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
