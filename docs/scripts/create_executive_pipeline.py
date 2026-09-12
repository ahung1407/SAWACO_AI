import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('figures', exist_ok=True)

def create_executive_pipeline():
    # Canvas chuẩn tỷ lệ ngang, thoáng đãng, tối giản
    fig, ax = plt.subplots(figsize=(15.0, 5.0), dpi=300)
    ax.axis('off')
    
    # 4 khối chiến lược cấp cao (High-Level)
    steps = [
        {
            "num": "1",
            "title": "Chuẩn bị Dữ liệu",
            "sub": "DATA PREPARATION",
            "desc": "Kết hợp dữ liệu mô phỏng\nvới ảnh chụp thực tế OV2640,\ntạo tập huấn luyện đa dạng",
            "color": "#0284C7",
            "bg": "#F0F9FF"
        },
        {
            "num": "2",
            "title": "Huấn luyện CNN",
            "sub": "MODEL TRAINING",
            "desc": "Huấn luyện mạng Modern CNN\nnhận dạng 10 lớp ký số (0-9),\ntích hợp chống học vẹt",
            "color": "#7C3AED",
            "bg": "#FAF5FF"
        },
        {
            "num": "3",
            "title": "Đóng gói Mô hình",
            "sub": "MODEL PACKAGING",
            "desc": "Xuất file Keras v3 chuẩn hóa,\ntối ưu dung lượng siêu nhẹ,\nsẵn sàng nạp vào bộ nhớ",
            "color": "#D97706",
            "bg": "#FFFBEB"
        },
        {
            "num": "4",
            "title": "Triển khai Dịch vụ AI",
            "sub": "API DEPLOYMENT",
            "desc": "Nạp RAM máy chủ FastAPI,\ncung cấp REST API suy luận\nthời gian thực cho Backend",
            "color": "#16A34A",
            "bg": "#F0FDF4"
        }
    ]
    
    x_start = 0.6
    card_w = 3.0
    card_h = 3.1
    gap = 0.75
    y_center = 2.1
    
    for i, step in enumerate(steps):
        x = x_start + i * (card_w + gap)
        y = y_center - card_h / 2
        
        # Thẻ bo góc chính (Card)
        rect = patches.FancyBboxPatch(
            (x, y), card_w, card_h,
            boxstyle="round,pad=0.08,rounding_size=0.18",
            facecolor=step["bg"],
            edgecolor=step["color"],
            linewidth=2.2,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Vòng tròn số thứ tự bước (Step Circle)
        circle = patches.Circle(
            (x + card_w / 2, y + card_h - 0.48), 0.32,
            facecolor=step["color"], edgecolor="none", zorder=3
        )
        ax.add_patch(circle)
        ax.text(x + card_w / 2, y + card_h - 0.48, step["num"],
                ha='center', va='center', fontsize=12.5, fontweight='bold', color='white', zorder=4)
        
        # Tiêu đề tiếng Việt
        ax.text(x + card_w / 2, y + card_h - 1.15, step["title"],
                ha='center', va='center', fontsize=12.5, fontweight='bold', color=step["color"], zorder=4)
        
        # Phụ đề tiếng Anh
        ax.text(x + card_w / 2, y + card_h - 1.48, step["sub"],
                ha='center', va='center', fontsize=8.2, fontweight='bold', color="#64748B", zorder=4)
        
        # Đường kẻ chia tách mờ
        ax.plot([x + 0.35, x + card_w - 0.35], [y + card_h - 1.7, y + card_h - 1.7],
                color=step["color"], linewidth=0.8, linestyle='--', alpha=0.4, zorder=3)
        
        # Mô tả ngắn gọn (Căn giữa hoàn hảo, khoảng cách thoáng)
        ax.text(x + card_w / 2, y + 0.68, step["desc"],
                ha='center', va='center', fontsize=9.2, linespacing=1.45, color="#334155", zorder=4)
        
        # Mũi tên liên kết giữa các thẻ
        if i < len(steps) - 1:
            arrow_x = x + card_w
            ax.annotate("",
                        xy=(arrow_x + gap - 0.08, y_center),
                        xytext=(arrow_x + 0.08, y_center),
                        arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=2.4, mutation_scale=18),
                        zorder=5)
            
    # Tiêu đề tổng quát phía trên
    ax.text(7.5, 4.35, "QUY TRÌNH TỔNG THỂ PHÁT TRIỂN VÀ TRIỂN KHAI PHÂN HỆ AI",
            ha='center', va='center', fontsize=14.5, fontweight='bold', color='#0F172A')
    
    ax.set_xlim(0, 15.0)
    ax.set_ylim(0.3, 4.8)
    
    out_file = 'figures/quy_trinh_huan_luyen_trien_khai.png'
    plt.savefig(out_file, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Generated clean high-level {out_file} successfully!")

if __name__ == '__main__':
    create_executive_pipeline()
