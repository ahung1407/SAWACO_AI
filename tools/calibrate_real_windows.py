import os, glob, json, cv2
import numpy as np

paths = sorted(glob.glob('real_data_base/*.jpg'))
boxes_dict = {}

for p in paths:
    fname = os.path.basename(p)
    img = cv2.imread(p)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 10)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in cnts if 0.4*h < cv2.boundingRect(c)[3] < 0.98*h and 0.8 < cv2.boundingRect(c)[3]/cv2.boundingRect(c)[2] < 3.5]
    
    kept = []
    for r in sorted(rects, key=lambda x: x[0]):
        if not any(max(r[0], k[0]) < min(r[0]+r[2], k[0]+k[2]) and (min(r[0]+r[2], k[0]+k[2]) - max(r[0], k[0]))/min(r[2], k[2]) > 0.3 for k in kept):
            kept.append(r)
            
    # Assign to the 5 physical window slots
    # Window 0: 150 - 250
    # Window 1: 330 - 430
    # Window 2: 510 - 610
    # Window 3: 720 - 820
    # Window 4: 920 - 1020
    assigned = {0: None, 1: None, 2: None, 3: None, 4: None}
    for r in kept:
        cx = r[0] + r[2]/2.0
        if 150 <= cx < 260 and assigned[0] is None:
            assigned[0] = cx
        elif 260 <= cx < 460 and assigned[1] is None:
            assigned[1] = cx
        elif 460 <= cx < 660 and assigned[2] is None:
            assigned[2] = cx
        elif 660 <= cx < 840 and assigned[3] is None:
            assigned[3] = cx
        elif cx >= 860 and assigned[4] is None:
            assigned[4] = cx

    # Determine reference spacing
    diffs = []
    if assigned[0] and assigned[1]: diffs.append(assigned[1] - assigned[0])
    if assigned[1] and assigned[2]: diffs.append(assigned[2] - assigned[1])
    spacing = float(np.median(diffs)) if diffs else 183.0

    # Base anchor cx0
    if assigned[0] is not None:
        cx0 = assigned[0]
    elif assigned[1] is not None:
        cx0 = assigned[1] - spacing
    else:
        cx0 = 200.0

    # Fill in centers
    c_centers = {}
    c_centers[0] = assigned[0] if assigned[0] is not None else cx0
    c_centers[1] = assigned[1] if assigned[1] is not None else (c_centers[0] + spacing)
    c_centers[2] = assigned[2] if assigned[2] is not None else (c_centers[1] + spacing)
    c_centers[3] = assigned[3] if assigned[3] is not None else (c_centers[2] + spacing * 1.22)
    c_centers[4] = assigned[4] if assigned[4] is not None else (c_centers[3] + spacing * 1.05)

    all_r = [r for r in kept if 0.4*h < r[3] < 0.98*h]
    avg_y = int(np.median([r[1] for r in all_r])) if all_r else int(h * 0.22)
    avg_h = int(np.median([r[3] for r in all_r])) if all_r else int(h * 0.55)

    y1 = max(0, int(avg_y - avg_h * 0.06))
    y2 = min(h, int(avg_y + avg_h * 1.06))
    bh = y2 - y1
    bw = 140

    final_boxes = []
    for idx in range(5):
        cx = c_centers[idx]
        x1 = max(0, min(w - bw, int(cx - bw / 2.0)))
        final_boxes.append([x1, y1, bw, bh])

    boxes_dict[fname] = final_boxes
    print(f"{fname}: centers={[int(c_centers[i]) for i in range(5)]}")

os.makedirs('data', exist_ok=True)
with open('data/real_window_boxes.json', 'w', encoding='utf-8') as f:
    json.dump(boxes_dict, f, indent=2)

print("Successfully generated calibrated data/real_window_boxes.json!")
