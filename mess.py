import requests
import random
import time

# URL của biểu mẫu Google Form
url = "https://docs.google.com/forms/d/e/1FAIpQLScfFvhO1P_e7e_A0mrJy8C7qI6r-cB3CaYNFLdAbPrn3skKew/formResponse"

# Các giá trị ẩn cần thiết của Google Form
# Các giá trị này được trích xuất từ HTML của biểu mẫu
hidden_fields = {
    "fbzx": "6180398345499205619",
    "pageHistory": "0", # Thường là '0' khi gửi toàn bộ form một lần
}

# --- Định nghĩa các tùy chọn cho từng câu hỏi ---

# 1. Thông tin người Khảo sát
age_options = ["1-18 tuổi", "19-22 tuổi", "23-30 tuổi", "30-45 tuổi", "Trên 45 tuổi"]
gender_options = ["Nam", "Nữ", "Khác"]
occupation_options = [
    "Học sinh", "Sinh viên", "Công nghệ thông tin / Lập trình viên",
    "Kỹ sư (cơ khí, điện, xây dựng,…)", "Giáo viên / Giảng viên",
    "Bác sĩ / Điều dưỡng / Y tế", "Nhân viên tài chính / kế toán / ngân hàng",
    "Nhân viên kinh doanh / bán hàng / marketing",
    "Nhân viên dịch vụ (nhà hàng, khách sạn, du lịch)", "Công nhân sản xuất",
    "Kinh doanh tự do / Chủ doanh nghiệp", "Freelancer / Làm việc tự do",
    "Nội trợ", "Đã nghỉ hưu"
]

# 2. Thói quen sử dụng ghế
sitting_hours_options = ["Dưới 2h", "2–4h", "4–6h", "Trên 6h"]
purpose_options = ["Làm việc văn phòng", "Học tập", "Giải trí (xem phim, chơi game, đọc sách)"] # Bỏ qua option ""
location_options = ["Văn phòng", "Nhà riêng", "Thư viện/quán café"] # Bỏ qua option ""
current_chair_type_options = ["Ghế văn phòng tiêu chuẩn", "Ghế công thái học (ergonomic)", "Ghế gỗ/nhựa cơ bản"] # Bỏ qua option ""
space_options = ["Rộng", "Trung bình", "Hẹp"]
table_type_options = ["Bàn làm việc tiêu chuẩn", "Bàn nâng hạ chiều cao (standing desk)", "Bàn ăn/bàn học cơ bản", "Chỉ ngồi ghế không sử dụng bàn"] # Bỏ qua option ""

# 3. Vấn đề & nhu cầu
sitting_problems_options = ["Đau lưng", "Đau cổ/ vai gáy", "Tê mỏi chân tay", "Không gặp vấn đề đáng kể"] # Bỏ qua option ""
accessories_options = ["Đệm lưng", "Đệm cổ", "Đệm ngồi", "Không"] # Bỏ qua option ""
importance_scale_options = ["1 – Không quan trọng", "2 – Ít quan trọng", "3 – Bình thường", "4 – Quan trọng", "5 – Rất quan trọng"]
ai_interest_options = ["Rất quan tâm", "Quan tâm ở mức vừa", "Không quan tâm lắm", "Không quan tâm"]

# 4. Tính năng kỳ vọng
ai_features_options = [
    "Tự động điều chỉnh độ cao phù hợp với chiều cao cơ thể",
    "Nhắc nhở thay đổi tư thế khi ngồi quá lâu",
    "Theo dõi và ghi lại thời gian ngồi/nghỉ",
    "Cá nhân hóa tư thế ngồi theo thói quen"
] # Bỏ qua option ""
app_connection_options = ["Có", "Không chắc", "Không"]

# 5. Khả năng chi trả & ý kiến
satisfaction_phrases = [
    "Ghế hiện tại của tôi khá thoải mái và phù hợp với không gian làm việc.",
    "Tôi hài lòng với thiết kế đơn giản và tính năng cơ bản của ghế.",
    "Chất liệu ghế tốt và bền, sử dụng lâu không hỏng hóc.",
    "Ghế có giá cả phải chăng và dễ dàng tìm mua."
]
discomfort_phrases = [
    "Ghế gây đau lưng sau khi ngồi lâu.",
    "Thiếu khả năng điều chỉnh độ cao và độ ngả lưng.",
    "Chất liệu không thoáng khí, gây khó chịu vào mùa hè.",
    "Kê tay không thoải mái và không thể điều chỉnh.",
    "Ghế quá cứng hoặc quá mềm."
]
premium_features_options = [
    "Massage nhẹ", "Sưởi ấm/làm mát", "Chất liệu da cao cấp",
    "Bảo hành dài hạn (3–5 năm)", "AI nâng cao (cá nhân hóa tư thế, theo dõi sức khỏe)"
]
replace_current_chair_options = ["Có", "Có nhưng tùy giá và chất lượng", "Không"]
dream_chair_phrases = [
    "Chiếc ghế trong mơ của tôi sẽ có khả năng tự động điều chỉnh hoàn toàn bằng AI, có chế độ massage và sưởi ấm.",
    "Tôi muốn một chiếc ghế có thiết kế hiện đại, vật liệu cao cấp, và ghi nhớ các tư thế yêu thích của tôi.",
    "Một chiếc ghế có thể phát hiện tư thế sai và nhắc nhở tôi điều chỉnh, đồng thời theo dõi sức khỏe tổng thể của tôi.",
    "Ghế có thể gấp gọn hoặc thay đổi hình dạng linh hoạt để phù hợp với nhiều mục đích sử dụng."
]

# --- Tạo dữ liệu ngẫu nhiên ---
def generate_random_data():
    data = hidden_fields.copy() # Bắt đầu với các trường ẩn

    # 1. Thông tin người Khảo sát
    data["entry.540194378"] = random.choice(age_options)
    data["entry.884164153"] = random.choice(gender_options)
    data["entry.783070196"] = random.choice(occupation_options)
    data["entry.666486714"] = str(random.randint(150, 190)) # Chiều cao từ 150-190 cm
    data["entry.510927030"] = str(random.randint(45, 100))  # Cân nặng từ 45-100 kg

    # 2. Thói quen sử dụng ghế
    data["entry.958687159"] = random.choice(sitting_hours_options)
    # Câu hỏi chọn nhiều đáp án (checkbox) - chọn ngẫu nhiên 1 đến tất cả các lựa chọn
    data["entry.1078234493"] = random.sample(purpose_options, random.randint(1, len(purpose_options)))
    data["entry.1360099246"] = random.sample(location_options, random.randint(1, len(location_options)))
    data["entry.809150247"] = random.choice(current_chair_type_options)
    data["entry.98016807"] = random.choice(space_options)
    data["entry.183591397"] = random.sample(table_type_options, random.randint(1, len(table_type_options)))

    # 3. Vấn đề & nhu cầu
    data["entry.1750770488"] = random.sample(sitting_problems_options, random.randint(1, len(sitting_problems_options)))
    data["entry.926408855"] = random.sample(accessories_options, random.randint(1, len(accessories_options)))
    
    # Mức độ quan trọng (scale 1-5)
    data["entry.310124769"] = random.choice(importance_scale_options) # Tính thoải mái
    data["entry.1272889944"] = random.choice(importance_scale_options) # Khả năng điều chỉnh
    data["entry.1937972216"] = random.choice(importance_scale_options) # Thiết kế/kiểu dáng
    data["entry.1800568741"] = random.choice(importance_scale_options) # Độ bền, chất lượng
    data["entry.327813654"] = random.choice(importance_scale_options) # Giá thành

    data["entry.609228551"] = random.choice(ai_interest_options)

    # 4. Tính năng kỳ vọng
    data["entry.1861174790"] = random.sample(ai_features_options, random.randint(1, len(ai_features_options)))
    data["entry.694823609"] = random.choice(app_connection_options)

    # 5. Khả năng chi trả & ý kiến
    data["entry.1341575167"] = random.choice(satisfaction_phrases)
    data["entry.578148391"] = random.choice(discomfort_phrases)
    data["entry.448461251"] = str(random.randint(2, 20)) # Giá từ 2-20 triệu VNĐ
    data["entry.1527556280"] = random.sample(premium_features_options, random.randint(1, len(premium_features_options)))
    data["entry.1272644157"] = random.choice(replace_current_chair_options)
    data["entry.1683124623"] = random.choice(dream_chair_phrases)

    return data

# --- Gửi biểu mẫu ---
def submit_form_randomly(num_submissions=1):
    for i in range(num_submissions):
        random_data = generate_random_data()
        print(f"--- Đang gửi lần {i+1}/{num_submissions} ---")
        # print("Dữ liệu gửi:", random_data) # Có thể bỏ comment để xem dữ liệu gửi

        try:
            res = requests.post(url, data=random_data)
            if res.status_code == 200:
                print(f"Lần {i+1}: Đã gửi thành công!")
            else:
                print(f"Lần {i+1}: Lỗi - Mã trạng thái HTTP: {res.status_code}")
                print(f"Phản hồi từ server:\n{res.text}")
        except requests.exceptions.RequestException as e:
            print(f"Lần {i+1}: Lỗi kết nối: {e}")

        # Tạm dừng một chút giữa các lần gửi để tránh bị chặn (nếu gửi nhiều)
        time.sleep(random.uniform(1, 3)) # Tạm dừng 1-3 giây

# --- Thực thi ---
if __name__ == "__main__":
    num_submissions = int(input("Bạn muốn gửi bao nhiêu lần (ví dụ: 10)? "))
    submit_form_randomly(num_submissions)