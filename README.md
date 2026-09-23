# HƯỚNG DẪN VẬN HÀNH SAWACO AI OCR & QUY CHUẨN DỰ ÁN

Hệ thống Microservice AI OCR nhận diện chỉ số đồng hồ nước cơ khí chuẩn **Tổng công ty Cấp nước Sài Gòn (Sawaco)**.
Mô hình chạy trên nền tảng **FastAPI (Python)** kết hợp mạng nơ-ron **Modern 3-Block CNN**, thuật toán phân đoạn lai (**Hybrid Contours + Grid Inference**) và cơ chế **Làm tròn sàn 2 chặng (Two-Stage Cascaded Floor Rule)**.

---

## 1. Cấu trúc Dự án & Quy chuẩn Lưu trữ (Directory Architecture)

Dự án được thiết lập quy tắc quản lý tệp đầu ra nghiêm ngặt để **tuyệt đối không làm ô nhiễm thư mục gốc (`root`)**:

```text
SAWACO_AI/
│
├── api_server.py              # FastAPI Server phục vụ API nhận diện AI OCR
├── inference_lib.py           # Thư viện lõi WaterMeterReader & Floor Rule
├── segmentation.py            # Thuật toán Hybrid Contours + Grid Inference
├── tunnel_manager.py          # Quản lý Tunnel tự động (Bore / Localtunnel)
├── water_meter_pipeline.py    # Kịch bản huấn luyện mạng Modern 3-Block CNN
├── water_meter_modern.keras   # Trọng số mô hình AI chính thức
├── image.png                  # Ảnh mẫu đồng hồ thực tế tham chiếu
├── requirements.txt           # Danh sách thư viện phụ thuộc
├── README.md                  # Tài liệu hướng dẫn vận hành & kiểm thử
│
├── data/                      # Dữ liệu kiểm thử chính thức
│   └── test_suites/           # 3 Tập kiểm thử độc lập (Suite 1, 2, 3 - 173 ảnh)
├── reports/                   # Nơi xuất báo cáo khoa học & benchmark
│   └── evaluation_benchmark_report.md
├── tools/                     # Bộ công cụ dòng lệnh (benchmark, retrain, inspect)
├── tests/                     # Các kịch bản test tự động
│   └── output/                # Nơi chứa ảnh tạm của tests (tự động dọn dẹp)
├── real_data_base/            # Kho ảnh chụp thực địa từ camera ESP32-S3
├── debug_images/              # Vết debug của API Server (tối đa 50 phiên gần nhất)
└── docs/                      # Toàn bộ tài liệu luận văn tốt nghiệp (LaTeX)
```

### Quy tắc sinh tệp (Output Rules):
1. **Không ghi ảnh ra thư mục gốc**: Mọi kịch bản sinh ảnh tạm, bounding box hoặc ảnh mô phỏng đều phải lưu vào `debug_images/`, `reports/`, hoặc `tests/output/`.
2. **Tự động dọn dẹp**: Các bài test tạm thời phải tự giải phóng tệp ảnh sau khi test xong. Thư mục `debug_images/` chỉ giữ tối đa 50 phiên gần nhất để bảo toàn dung lượng đĩa.
3. **Cơ chế `.gitignore` 2 lớp**: Mọi tệp ảnh phát sinh ngoài ý muốn ở thư mục gốc đều bị chặn tự động, không bao giờ lọt vào Git commit.

---

## 2. Danh mục Các Bài Test & Hướng dẫn Thực hiện

Hệ thống cung cấp sẵn các kịch bản kiểm thử độc lập, có quy hoạch rõ ràng về mục đích và nơi xuất kết quả:

| Bài Test | Lệnh thực hiện | Mục đích kiểm thử | Kết quả xuất ra ở đâu? |
|---|---|---|---|
| **1. Benchmark 3 Tập Kiểm thử (173 ảnh)** | `python tools/benchmark_suites.py` | Đo đạc độ chính xác toàn diện (Suite 1, 2, 3), tính ma trận nhầm lẫn 10x10, độ trễ và FPS. | Xuất bảng tổng kết trên terminal và ghi file báo cáo tại `reports/evaluation_benchmark_report.md`. |
| **2. Đọc thử 1 ảnh mặt đồng hồ bất kỳ** | `python tools/read_meter_box.py <đường_dẫn_ảnh>` | Kiểm tra nhanh khả năng cắt 5 ô số và đọc chỉ số của 1 ảnh cụ thể (ví dụ: `image.png`). | In kết quả chi tiết lên màn hình; hình vẽ bounding box lưu vào `debug_images/debug_final_boxes.jpg`. |
| **3. Kiểm thử Cảnh báo Dị vật (NaN Gate)** | `python tests/test_nan_simulation.py` | Giả lập các tình huống mặt kính bị lá cây, bùn đất, vết ố, chóa đèn flash che khuất để kiểm tra trạm từ chối `NaN`. | Ảnh mô phỏng lưu vào `test_images/nan_sim/` (đã cấu hình bỏ qua trong Git). |
| **4. Kiểm thử Ký số Cơ học 0-9** | `python tests/test_water_meter_digits.py` | Kiểm thử độ chính xác nhận diện nhanh trên cả 10 chữ số từ 0 đến 9. | Tạm lưu trong `tests/output/` và **tự động xóa sạch ngay sau khi test xong**. |
| **5. Kiểm thử Huấn luyện lại Model** | `python tools/retrain_with_real_data.py` | Tinh chỉnh (fine-tuning) mô hình với dữ liệu thực tế mới khi cần cập nhật weights. | Xuất file trọng số `water_meter_modern.keras`. |

---

## 3. Cách Khởi Chạy Server AI OCR (Production)

### Bước 1: Cài đặt thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### Bước 2: Khởi chạy Server
Chạy trực tiếp kịch bản máy chủ:
```bash
python api_server.py
```
*(Hoặc chạy qua Uvicorn: `uvicorn api_server:app --host 0.0.0.0 --port 8000`)*

Khi khởi động thành công, server sẽ:
1. Tải sẵn mô hình `water_meter_modern.keras` vào RAM để đạt độ trễ suy luận siêu tốc (20-50 ms).
2. Tự động thiết lập Tunnel mạng (Bore/Localtunnel) để Backend Spring Boot hoặc thiết bị biên từ xa có thể kết nối ngay lập tức.
3. Lắng nghe tại cổng `http://localhost:8000`.

---

## 4. Đặc tả API Endpoint (`POST /api/ai/ocr`)

* **URL**: `http://localhost:8000/api/ai/ocr`
* **Content-Type**: `multipart/form-data`
* **Tham số**: `file` (tệp hình ảnh đồng hồ)

### Ví dụ gọi bằng cURL:
```bash
curl -X POST "http://localhost:8000/api/ai/ocr" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@image.png"
```

### Ví dụ gọi bằng Python (`requests`):
```python
import requests

url = "http://localhost:8000/api/ai/ocr"
with open("image.png", "rb") as f:
    response = requests.post(url, files={"file": f})

print("Kết quả:", response.json())
```

---

## 5. Cấu trúc JSON Phản hồi theo Chuẩn Nghiệp vụ Sawaco

### A. Trường hợp nhận diện thành công:
```json
{
  "status": "success",
  "billing_m3": 28,
  "billing_m3_str": "0028",
  "fraction_liters": 100,
  "formatted": "0028.1 m3",
  "water_reading": 28.1,
  "raw_string": "00281",
  "message": "Recognition successful"
}
```
* `billing_m3`: Chỉ số nguyên mét khối ($m^3$) trích xuất từ **4 bánh xe màu đen** ($D_0 D_1 D_2 D_3$). Đây là đại lượng pháp lý chính thức để xuất hóa đơn cước nước.
* `fraction_liters`: Chỉ số phụ từ **bánh xe màu đỏ thứ 5** ($D_4 \times 100\text{ lít}$), phục vụ giám sát kỹ thuật vi mô và phát hiện rò rỉ.
* `formatted`: Chuỗi định dạng trực quan (ví dụ: `"0028.1 m3"`).

### B. Trường hợp bị lá cây / bùn đất che khuất:
```json
{
  "status": "warning",
  "message": "Digits obscured by dirt or foreign objects. Please clean and recapture.",
  "billing_m3": 0,
  "water_reading": 0.0
}
```
* Backend nhận diện mã này để thông báo cho nhân viên kiểm tra, lau kính và chụp lại ảnh nhằm bảo vệ quyền lợi người tiêu dùng.

---

## 6. Vết Debug của Server (`debug_images/`)

Mỗi yêu cầu nhận diện gửi đến API Server sẽ tự động lưu lại phiên xử lý vào `debug_images/latest/` gồm:
* `0_original.jpg`: Ảnh chụp nguyên bản gửi lên.
* `0_boxes.jpg`: Ảnh trực quan 5 bounding boxes đã được phân đoạn lai và gọt sạch viền nhựa $22\%$.
* `1_digit_1.jpg` đến `1_digit_5.jpg`: 5 ảnh chữ số đơn lẻ kích thước $28 \times 28$.
* `result.json`: Log toàn bộ xác suất Softmax và độ tin cậy của từng ô số.

*(Hệ thống tự động xoay vòng dọn dẹp, chỉ giữ lại 50 phiên gần nhất, không bao giờ làm đầy ổ đĩa).*
