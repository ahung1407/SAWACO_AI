import os
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from inference_lib import WaterMeterReader
from segmentation import segment_meter_digits

os.makedirs('figures', exist_ok=True)

# =====================================================================
# 1. ĐO LƯỜNG ĐỘ CHÍNH XÁC VÀ MA TRẬN NHẦM LẪN (ACCURACY & CONFUSION MATRIX)
# =====================================================================
def evaluate_classification(reader, test_dir='datasets/test'):
    print("=" * 65)
    print("1. ĐÁNH GIÁ ĐỘ CHÍNH XÁC MÔ HÌNH TRÊN TẬP TEST")
    print("=" * 65)
    
    classes = [str(i) for i in range(10)]
    confusion_matrix = np.zeros((10, 10), dtype=int)
    
    total_samples = 0
    correct_samples = 0
    
    start_time = time.time()
    
    for true_label in range(10):
        class_folder = os.path.join(test_dir, str(true_label))
        if not os.path.exists(class_folder):
            continue
            
        img_names = [f for f in os.listdir(class_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        for name in img_names:
            img_path = os.path.join(class_folder, name)
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Dự đoán ký số
            res = reader.predict(img)
            pred_label = res["digit"]
            if pred_label == 'NaN':
                continue
            pred_label = int(pred_label)
            
            confusion_matrix[true_label, pred_label] += 1
            total_samples += 1
            if pred_label == true_label:
                correct_samples += 1
                
    elapsed = time.time() - start_time
    overall_acc = correct_samples / total_samples if total_samples > 0 else 0
    
    print(f"Tổng số mẫu kiểm thử: {total_samples}")
    print(f"Số mẫu đoán đúng:     {correct_samples}")
    print(f"Độ chính xác (Accuracy): {overall_acc * 100:.2f}%\n")
    
    # In bảng chi tiết từng lớp
    print(f"{'Ký số':<8}{'Số mẫu':<10}{'Precision':<12}{'Recall':<12}{'F1-Score':<10}")
    print("-" * 52)
    
    f1_list = []
    for c in range(10):
        tp = confusion_matrix[c, c]
        fp = np.sum(confusion_matrix[:, c]) - tp
        fn = np.sum(confusion_matrix[c, :]) - tp
        total_c = np.sum(confusion_matrix[c, :])
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_list.append(f1)
        
        print(f"Số {c:<5}{total_c:<10}{precision*100:>6.2f}%    {recall*100:>6.2f}%    {f1*100:>6.2f}%")
        
    macro_f1 = np.mean(f1_list)
    print("-" * 52)
    print(f"Macro F1-Score: {macro_f1 * 100:.2f}%\n")
    
    # Vẽ Ma trận nhầm lẫn (Heatmap)
    plot_confusion_matrix(confusion_matrix)
    return overall_acc, confusion_matrix

def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    
    # Chuẩn hóa về tỷ lệ %
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    cax = ax.matshow(cm_norm, cmap='Blues', alpha=0.85)
    fig.colorbar(cax)
    
    for i in range(10):
        for j in range(10):
            count = cm[i, j]
            pct = cm_norm[i, j] * 100
            text = f"{count}\n({pct:.1f}%)" if count > 0 else "0"
            color = "white" if cm_norm[i, j] > 0.5 else "#1E293B"
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8.5, fontweight='medium')
            
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(range(10), fontsize=11, fontweight='bold')
    ax.set_yticklabels(range(10), fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Nhãn dự đoán (Predicted Label)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Nhãn thực tế (True Label)', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title('Ma trận nhầm lẫn (Confusion Matrix) nhận dạng ký số 0 - 9', fontsize=13, fontweight='bold', pad=18)
    
    out_path = 'figures/confusion_matrix.png'
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Đã lưu biểu đồ: {out_path}")

# =====================================================================
# 2. ĐO LƯỜNG THỜI GIAN VÀ ĐỘ TRỄ (LATENCY BENCHMARK)
# =====================================================================
def benchmark_latency(reader, n_iterations=100):
    print("\n" + "=" * 65)
    print("2. ĐO LƯỜNG HIỆU NĂNG THỜI GIAN (LATENCY BENCHMARK)")
    print("=" * 65)
    
    dummy_digit = np.full((100, 100, 3), 240, dtype=np.uint8)
    cv2.putText(dummy_digit, "5", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (20, 20, 20), 6)
    
    dummy_meter = np.full((120, 480, 3), 200, dtype=np.uint8)
    
    # 2.1 Tiền xử lý (Min-RGB + CLAHE + Resize 28x28)
    t_pre = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = reader.preprocess(dummy_digit)
        t_pre.append((time.perf_counter() - t0) * 1000)
        
    # 2.2 Suy luận CNN (5 ký số song song)
    batch_tensor = np.zeros((5, 28, 28, 1), dtype=np.float32)
    t_cnn = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = reader.model(batch_tensor, training=False)
        t_cnn.append((time.perf_counter() - t0) * 1000)
        
    # 2.3 Phân đoạn ảnh (Segmentation 5 ô số)
    t_seg = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        _ = segment_meter_digits(dummy_meter, num_digits=5, margin_ratio=0.22)
        t_seg.append((time.perf_counter() - t0) * 1000)
        
    # 2.4 Hậu xử lý logic bánh răng cơ khí (5 ký số)
    dummy_probs = np.full((5, 10), 0.05, dtype=np.float32)
    dummy_probs[:, 5] = 0.5
    t_logic = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        for p in dummy_probs:
            _ = reader.smart_read_logic(p)
        t_logic.append((time.perf_counter() - t0) * 1000)
        
    # In kết quả dạng bảng khoa học
    print(f"{'Công đoạn xử lý':<35}{'Trung bình (ms)':<16}{'Độ lệch chuẩn':<15}{'Min - Max (ms)'}")
    print("-" * 75)
    
    tasks = [
        ("Tiền xử lý (Min-RGB, CLAHE, 28x28)", t_pre),
        ("Phân đoạn 5 ô số (Segmentation)", t_seg),
        ("Suy luận Modern CNN (5 ký số)", t_cnn),
        ("Kiểm tra logic cơ khí (Rolling)", t_logic),
    ]
    
    total_mean = 0
    for name, times in tasks:
        mean_v = np.mean(times)
        std_v = np.std(times)
        min_v = np.min(times)
        max_v = np.max(times)
        total_mean += mean_v
        print(f"{name:<35}{mean_v:>8.2f} ms      ±{std_v:>6.2f} ms     {min_v:.2f} - {max_v:.2f} ms")
        
    print("-" * 75)
    print(f"{'TỔNG THỜI GIAN SUY LUẬN (End-to-End)':<35}{total_mean:>8.2f} ms\n")

if __name__ == '__main__':
    reader = WaterMeterReader('water_meter_modern.keras')
    evaluate_classification(reader)
    benchmark_latency(reader, n_iterations=50)
