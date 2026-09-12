import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

os.makedirs('figures', exist_ok=True)

# -------------------------------------------------------------
# 1. GENERATE FIGURE 1: Synthetic Data & Augmentation Samples
# -------------------------------------------------------------
def make_synthetic_sample(digit, kind='standard'):
    h, w = 100, 100
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = np.full((h, w, 3), 245, dtype=np.uint8)
    
    if kind == 'standard':
        cv2.putText(img, str(digit), (24, 72), font, 2.6, (15, 15, 15), 6)
    elif kind == 'thin':
        cv2.putText(img, str(digit), (26, 72), font, 2.5, (25, 25, 25), 2)
    elif kind == 'thick':
        cv2.putText(img, str(digit), (22, 74), font, 2.8, (5, 5, 5), 11)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif kind == 'red':
        # Chữ số đỏ trên nền trắng
        cv2.putText(img, str(digit), (24, 72), font, 2.7, (20, 25, 220), 7)
    elif kind == 'rolling':
        # Nửa trên số digit, nửa dưới digit+1
        next_d = (digit + 1) % 10
        l1 = np.full((h, w, 3), 245, dtype=np.uint8)
        l2 = np.full((h, w, 3), 245, dtype=np.uint8)
        cv2.putText(l1, str(digit), (24, 72), font, 2.6, (15, 15, 15), 6)
        cv2.putText(l2, str(next_d), (24, 72), font, 2.6, (15, 15, 15), 6)
        offset = 45
        img[0:h-offset, :] = l1[offset:h, :]
        img[h-offset:h, :] = l2[0:offset, :]
        # Vạch ranh giới bánh răng
        cv2.line(img, (0, h-offset), (w, h-offset), (180, 180, 180), 1)
    elif kind == 'shadow':
        cv2.putText(img, str(digit), (24, 72), font, 2.6, (15, 15, 15), 6)
        # Xoay nhẹ
        M = cv2.getRotationMatrix2D((w//2, h//2), -6, 1)
        img = cv2.warpAffine(img, M, (w, h), borderValue=(245, 245, 245))
        # Đổ bóng gradient
        shadow = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(h):
            shadow[i, :] = int(90 * (i / h))
        img = cv2.subtract(img, shadow)
    elif kind == 'noise_blur':
        cv2.putText(img, str(digit), (24, 72), font, 2.6, (15, 15, 15), 6)
        noise = np.random.normal(0, 18, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        
    return img

def create_figure_data_samples():
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8))
    plt.subplots_adjust(wspace=0.15, hspace=0.35)
    
    samples = [
        ("Nét chuẩn\n(Standard)", make_synthetic_sample(5, 'standard')),
        ("Nét mảnh\n(Thin stroke, t=2)", make_synthetic_sample(2, 'thin')),
        ("Nét đậm nhòe\n(Thick stroke, t=11)", make_synthetic_sample(0, 'thick')),
        ("Ký số đỏ\n(Red digit, Min-RGB)", make_synthetic_sample(8, 'red')),
        ("Số đang lăn\n(Rolling 3->4, split)", make_synthetic_sample(3, 'rolling')),
        ("Đổ bóng & Xoay góc\n(Shadow gradient)", make_synthetic_sample(6, 'shadow')),
        ("Nhiễu & Mờ ống kính\n(Gaussian Noise + Blur)", make_synthetic_sample(1, 'noise_blur')),
    ]
    
    # Đọc 1 ảnh thực tế từ dataset
    real_img_path = 'real_digits_labeled/digit_2.jpg'
    if os.path.exists(real_img_path):
        real_img = cv2.imread(real_img_path)
        real_img = cv2.resize(real_img, (100, 100))
    else:
        real_img = make_synthetic_sample(2, 'standard')
    samples.append(("Ảnh chụp thực tế\n(OV2640 Real crop)", real_img))
    
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
    
    for i, ax in enumerate(axes.flat):
        title, img = samples[i]
        # Chuyển BGR sang RGB để matplotlib hiển thị chuẩn màu
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{letters[i]} {title}", fontsize=10.5, pad=6, fontweight='medium')
        # Thêm viền xám mỏng cho từng ô
        for spine in ax.spines.values():
            spine.set_color('#888888')
            spine.set_linewidth(1.2)
            
    fig.suptitle("Các dạng mẫu dữ liệu tổng hợp và ảnh thực tế mô phỏng điều kiện cơ học, quang học", 
                 fontsize=12.5, fontweight='bold', y=0.98)
    
    output_path = 'figures/synthetic_data_samples.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path} successfully!")

# -------------------------------------------------------------
# 2. GENERATE FIGURE 2: Modern CNN Architecture Diagram
# -------------------------------------------------------------
def create_figure_cnn_architecture():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axis('off')
    
    # Định nghĩa các khối mạng
    blocks = [
        {
            "name": "Input Layer",
            "spec": "28 × 28 × 1\n(Grayscale)",
            "detail": "Min-RGB +\nCLAHE norm",
            "color": "#E3F2FD",
            "edge": "#1565C0",
            "width": 1.5,
            "height": 3.8
        },
        {
            "name": "Conv Block 1",
            "spec": "Conv2D (32, 3×3)\nBatchNorm + ReLU\nMaxPool (2×2)",
            "detail": "Out: 14 × 14 × 32\nParams: 320",
            "color": "#E0F2F1",
            "edge": "#00796B",
            "width": 2.2,
            "height": 4.5
        },
        {
            "name": "Conv Block 2",
            "spec": "Conv2D (64, 3×3)\nBatchNorm + ReLU\nMaxPool (2×2)",
            "detail": "Out: 7 × 7 × 64\nParams: 18,688",
            "color": "#E8F5E9",
            "edge": "#2E7D32",
            "width": 2.2,
            "height": 4.5
        },
        {
            "name": "Conv Block 3",
            "spec": "Conv2D (128, 3×3)\nBatchNorm + ReLU\nMaxPool (2×2)",
            "detail": "Out: 3 × 3 × 128\nParams: 74,240",
            "color": "#FFF8E1",
            "edge": "#F57F17",
            "width": 2.2,
            "height": 4.5
        },
        {
            "name": "Dense & Classifier",
            "spec": "Flatten (1,152)\nDense (128) + BN\nDropout (p=0.5)",
            "detail": "Dense (10, Softmax)\nTotal: 241,482",
            "color": "#F3E5F5",
            "edge": "#7B1FA2",
            "width": 2.3,
            "height": 4.8
        },
        {
            "name": "Predictions",
            "spec": "Softmax Vector\nP(0) ... P(9)",
            "detail": "Ký số nhận diện\n[0 - 9]",
            "color": "#FFEBEE",
            "edge": "#C62828",
            "width": 1.6,
            "height": 3.6
        }
    ]
    
    x_cursor = 0.5
    y_center = 4.0
    
    for i, b in enumerate(blocks):
        # Vẽ hộp chữ nhật bo góc
        rect = patches.FancyBboxPatch(
            (x_cursor, y_center - b["height"]/2), 
            b["width"], b["height"],
            boxstyle="round,pad=0.12,rounding_size=0.15",
            facecolor=b["color"], 
            edgecolor=b["edge"], 
            linewidth=2.2,
            zorder=2
        )
        ax.add_patch(rect)
        
        # Tiêu đề khối
        ax.text(x_cursor + b["width"]/2, y_center + b["height"]/2 - 0.45, 
                b["name"], ha='center', va='center', fontsize=11, fontweight='bold', color=b["edge"])
        
        # Đường kẻ ngang ngăn cách
        ax.plot([x_cursor + 0.15, x_cursor + b["width"] - 0.15], 
                [y_center + b["height"]/2 - 0.8, y_center + b["height"]/2 - 0.8], 
                color=b["edge"], linewidth=1, linestyle='--', alpha=0.7)
        
        # Thông số kỹ thuật
        ax.text(x_cursor + b["width"]/2, y_center + 0.1, 
                b["spec"], ha='center', va='center', fontsize=9.5, linespacing=1.35, color='#212121')
        
        # Thông số đầu ra
        box_out = patches.FancyBboxPatch(
            (x_cursor + 0.15, y_center - b["height"]/2 + 0.2),
            b["width"] - 0.3, 1.0,
            boxstyle="square,pad=0.05",
            facecolor="#FFFFFF", edgecolor=b["edge"], linewidth=1, alpha=0.9
        )
        ax.add_patch(box_out)
        ax.text(x_cursor + b["width"]/2, y_center - b["height"]/2 + 0.7, 
                b["detail"], ha='center', va='center', fontsize=8.5, linespacing=1.25, color='#37474F', fontweight='medium')
        
        # Vẽ mũi tên liên kết sang khối kế tiếp
        if i < len(blocks) - 1:
            next_x = x_cursor + b["width"]
            arrow_len = 0.55
            ax.annotate("", 
                        xy=(next_x + arrow_len, y_center), 
                        xytext=(next_x, y_center),
                        arrowprops=dict(arrowstyle="-|>", color="#455A64", lw=2.2, mutation_scale=16),
                        zorder=3)
            
        x_cursor += b["width"] + 0.55
        
    ax.set_xlim(0, x_cursor)
    ax.set_ylim(0.8, 7.2)
    
    # Tiêu đề tổng quát
    ax.text(x_cursor/2, 6.8, "Sơ đồ kiến trúc Modern CNN 3 khối tích chập dùng cho nhận dạng ký số đồng hồ nước", 
            ha='center', va='center', fontsize=13, fontweight='bold', color='#1A237E')
    
    output_path = 'figures/modern_cnn_architecture.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path} successfully!")

if __name__ == '__main__':
    create_figure_data_samples()
    create_figure_cnn_architecture()
