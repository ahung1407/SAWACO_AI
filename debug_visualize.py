import os
import cv2
from segmentation import segment_meter_digits

def run_debug(img_path, out_dir):
    if not os.path.exists(img_path):
        print(f"File {img_path} not found.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    digits = segment_meter_digits(img_path, num_digits=5, margin_ratio=0.15)
    
    md_content = f"# Debug Cropped Images for {img_path}\n\n"
    
    for i, d in enumerate(digits):
        filename = f"digit_{i}.jpg"
        filepath = os.path.join(out_dir, filename)
        cv2.imwrite(filepath, d)
        abs_path = os.path.abspath(filepath)
        # Markdown requires forward slashes for URLs
        abs_path = abs_path.replace("\\", "/")
        md_content += f"## Digit {i+1}\n![Digit {i+1}](file:///{abs_path})\n\n"
        
    with open(f"{out_dir}_visuals.md", "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Debug visuals generated in {out_dir}_visuals.md")

if __name__ == "__main__":
    run_debug("test_meter.jpg", "debug_crops_cli_test")
    run_debug("debug_images/0_anh_goc_tu_java.jpg", "debug_crops_cli_java")
