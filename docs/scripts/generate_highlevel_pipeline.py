import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('figures', exist_ok=True)

def create_highlevel_pipeline_figure():
    fig, ax = plt.subplots(figsize=(16.0, 7.8), dpi=300)
    ax.axis('off')
    
    stages = [
        {
            "phase": "BƯỚC 1",
            "title": "Nguồn & Sinh dữ liệu",
            "badge": "DATA INGESTION",
            "badge_color": "#0284C7",
            "bg_color": "#F8FAFC",
            "border_color": "#0284C7",
            "x": 0.6,
            "width": 3.35,
            "items": [
                ("1.1 Sinh dữ liệu tổng hợp", "#0369A1", True),
                ("• Mô phỏng số lăn lấp lửng (Rolling)\n• Mô phỏng chữ số màu đỏ (Min-RGB)\n• Biến thiên nét chữ (t = 2 đến 12)\n• Đổ bóng râm, nhiễu và làm mờ", "#334155", False),
                ("1.2 Thu thập ảnh thực tế", "#0369A1", True),
                ("• Trích xuất ô số từ camera OV2640\n• Tăng cường dữ liệu (Augmentation)", "#334155", False),
            ]
        },
        {
            "phase": "BƯỚC 2",
            "title": "Hợp nhất & Tiền xử lý",
            "badge": "PREPROCESSING",
            "badge_color": "#0D9488",
            "bg_color": "#F8FAFC",
            "border_color": "#0D9488",
            "x": 4.45,
            "width": 3.35,
            "items": [
                ("2.1 Hợp nhất tập dữ liệu", "#0F766E", True),
                ("• Kết hợp > 8.000 mẫu số đa dạng\n• Bao phủ đầy đủ 10 lớp ký số (0..9)\n• Cân bằng tỷ lệ mẫu giữa các lớp", "#334155", False),
                ("2.2 Chuẩn hóa đầu vào (Input)", "#0F766E", True),
                ("• Chuyển ảnh xám bằng lọc Min-RGB\n• Cân bằng sáng thích ứng CLAHE\n• Đưa về kích thước chuẩn 28x28x1", "#334155", False),
            ]
        },
        {
            "phase": "BƯỚC 3",
            "title": "Huấn luyện Modern CNN",
            "badge": "MODEL TRAINING",
            "badge_color": "#7C3AED",
            "bg_color": "#F8FAFC",
            "border_color": "#7C3AED",
            "x": 8.3,
            "width": 3.35,
            "items": [
                ("3.1 Cấu trúc Modern CNN", "#6D28D9", True),
                ("• 3 khối Conv2D (32, 64, 128 bộ lọc)\n• Tích hợp BatchNorm và kích hoạt ReLU\n• MaxPooling2D (2x2) giảm chiều", "#334155", False),
                ("3.2 Tối ưu & Chống quá khớp", "#6D28D9", True),
                ("• Tầng Dense 128 + Dropout (p = 0.5)\n• Tối ưu Adam, Crossentropy, Batch 32\n• Huấn luyện hội tụ qua 15 Epochs", "#334155", False),
            ]
        },
        {
            "phase": "BƯỚC 4",
            "title": "Đóng gói & Triển khai",
            "badge": "DEPLOYMENT & API",
            "badge_color": "#16A34A",
            "bg_color": "#F8FAFC",
            "border_color": "#16A34A",
            "x": 12.15,
            "width": 3.35,
            "items": [
                ("4.1 Đóng gói mô hình .keras", "#15803D", True),
                ("• Xuất định dạng chuẩn Keras v3\n• File: water_meter_modern.keras\n• Kích thước gọn nhẹ (< 3 MB)", "#334155", False),
                ("4.2 Dịch vụ FastAPI trên RAM", "#15803D", True),
                ("• Tải sẵn vào RAM lúc khởi động\n• Nhận ảnh từ Spring Boot Backend\n• Tốc độ suy luận siêu tốc: 20 - 50 ms", "#334155", False),
            ]
        }
    ]
    
    y_top = 6.4
    card_h = 5.2
    
    for i, st in enumerate(stages):
        x = st["x"]
        w = st["width"]
        
        # Hộp card nền chính
        card = patches.FancyBboxPatch(
            (x, y_top - card_h), w, card_h,
            boxstyle="round,pad=0.1,rounding_size=0.18",
            facecolor=st["bg_color"],
            edgecolor=st["border_color"],
            linewidth=2.2,
            zorder=2
        )
        ax.add_patch(card)
        
        # Header banner của card
        header_banner = patches.FancyBboxPatch(
            (x + 0.1, y_top - 1.05), w - 0.2, 0.95,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=st["border_color"],
            edgecolor="none",
            alpha=0.12,
            zorder=3
        )
        ax.add_patch(header_banner)
        
        # Badge nhỏ định danh giai đoạn
        badge = patches.FancyBboxPatch(
            (x + 0.25, y_top - 0.42), w - 0.5, 0.32,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=st["badge_color"],
            edgecolor="none",
            zorder=4
        )
        ax.add_patch(badge)
        ax.text(x + w/2, y_top - 0.26, f"{st['phase']}: {st['badge']}", 
                ha='center', va='center', fontsize=9.2, fontweight='bold', color='white', zorder=5)
        
        # Tiêu đề giai đoạn
        ax.text(x + w/2, y_top - 0.75, st["title"], 
                ha='center', va='center', fontsize=11.5, fontweight='bold', color=st["border_color"], zorder=5)
        
        # Đường kẻ ngang ngăn cách
        ax.plot([x + 0.2, x + w - 0.2], [y_top - 1.18, y_top - 1.18], 
                color=st["border_color"], linewidth=1.2, linestyle='--', alpha=0.45, zorder=3)
        
        # Vẽ các mục nội dung bên trong
        curr_y = y_top - 1.55
        for text, col, is_subhead in st["items"]:
            if is_subhead:
                # Nền thẻ mục con
                sub_bg = patches.FancyBboxPatch(
                    (x + 0.16, curr_y - 0.16), w - 0.32, 0.32,
                    boxstyle="round,pad=0.03,rounding_size=0.06",
                    facecolor="#FFFFFF", edgecolor=st["border_color"], linewidth=1.0, alpha=0.95, zorder=3
                )
                ax.add_patch(sub_bg)
                ax.text(x + 0.26, curr_y, text, 
                        ha='left', va='center', fontsize=9.2, fontweight='bold', color=col, zorder=4)
                curr_y -= 0.38
            else:
                ax.text(x + 0.26, curr_y, text, 
                        ha='left', va='top', fontsize=8.8, linespacing=1.4, color=col, zorder=4)
                num_lines = text.count('\n') + 1
                curr_y -= (num_lines * 0.32 + 0.22)
                
        # Mũi tên liên kết giữa các giai đoạn
        if i < len(stages) - 1:
            arrow_start_x = x + w
            arrow_end_x = stages[i+1]["x"]
            arrow_y = y_top - card_h / 2
            ax.annotate("", 
                        xy=(arrow_end_x + 0.04, arrow_y), 
                        xytext=(arrow_start_x - 0.04, arrow_y),
                        arrowprops=dict(arrowstyle="-|>", color="#334155", lw=2.8, mutation_scale=20),
                        zorder=5)
            
    # Tiêu đề tổng quát toàn bộ sơ đồ
    ax.text(8.0, 7.35, "QUY TRÌNH TỔNG THỂ HUẤN LUYỆN VÀ TRIỂN KHAI MÔ HÌNH AI RECOGNITION", 
            ha='center', va='center', fontsize=14.5, fontweight='bold', color='#0F172A')
    ax.text(8.0, 7.02, "(Chiến lược kết hợp dữ liệu tổng hợp, dữ liệu thực tế OV2640, huấn luyện Modern CNN và đóng gói REST API)", 
            ha='center', va='center', fontsize=10.2, fontstyle='italic', color='#64748B')
            
    ax.set_xlim(0, 16.1)
    ax.set_ylim(0.7, 7.6)
    
    out_file = 'figures/quy_trinh_huan_luyen_trien_khai.png'
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"Generated {out_file} successfully!")

if __name__ == '__main__':
    create_highlevel_pipeline_figure()
