import cv2
import pytesseract
import numpy as np
import os

import clear_images
def find_and_read_meter(image_path, debug_folder=None, expected_len=5):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Cannot read image from {image_path}")
        return None, None, None

    output_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- STEP 1: FIND THE LARGE CIRCLE (METER FACE) ---
    blurred = cv2.medianBlur(gray, 5)
    rows = blurred.shape[0]
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=rows / 8,
        param1=100, param2=30,
        minRadius=int(rows / 4), maxRadius=int(rows / 2)
    )

    if circles is None:
        print("Failed to automatically detect water meter in image.")
        return None, image, None

    circles = np.uint16(np.around(circles))
    c = circles[0, 0]
    center_x, center_y, radius = c[0], c[1], c[2]
    cv2.circle(output_image, (center_x, center_y), radius, (0, 255, 0), 3)

    # --- STEP 2: CREATE A SEARCH REGION FOR THE DIGITS ---
    y_start = center_y - int(radius * 0.4)
    y_end   = center_y - int(radius * 0.2)
    x_start = center_x - int(radius * 0.4)
    x_end   = center_x + int(radius * 0.3)

    y_start = max(0, y_start)
    y_end   = min(gray.shape[0], y_end)
    x_start = max(0, x_start)
    x_end   = min(gray.shape[1], x_end)

    search_zone = gray[y_start:y_end, x_start:x_end]
    cv2.rectangle(output_image, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)

    if search_zone.size == 0:
        print("Search zone is empty.")
        return None, output_image, None

    # --- STEP 3: PREPROCESSING FOR OCR ---
    blur = cv2.GaussianBlur(search_zone, (5, 5), 0)
    _, ocr_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small noise
    contours, _ = cv2.findContours(ocr_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 20
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            cv2.drawContours(ocr_thresh, [cnt], -1, 0, -1)

    # --- STEP 4: Resize and morphology ---
    scale = 2
    ocr_input = cv2.resize(ocr_thresh, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((2, 2), np.uint8)
    ocr_input = cv2.morphologyEx(ocr_input, cv2.MORPH_CLOSE, kernel)

    # Save debug image if needed
    if debug_folder:
        os.makedirs(debug_folder, exist_ok=True)
        debug_path = os.path.join(debug_folder, f"debug_{os.path.basename(image_path)}")
        cv2.imwrite(debug_path, ocr_input)
        print(f"Debug image saved to {debug_path}")

    # --- STEP 5: OCR with multiple PSM modes ---
    psm_modes = [10]
    best_digits = ""
    for psm in psm_modes:
        config = f'--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(ocr_input, config=config)
        digits = ''.join(filter(str.isdigit, text))
        if len(digits) > len(best_digits):
            best_digits = digits

    # --- STEP 6: Normalize result length ---
    if best_digits:
        if len(best_digits) > expected_len:
            best_digits = best_digits[:expected_len]
        elif len(best_digits) < expected_len:
            best_digits = best_digits.rjust(expected_len, "1")  # pad bằng '1' hoặc '0'

    print(f"OCR candidates -> Final extracted: '{best_digits}'")

    return (best_digits if best_digits else None), output_image, ocr_input


# --- MAIN PROGRAM WITH TEST CASES ---
def process_images(path, output_folder="results", debug_folder="debug", test_cases=None):
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(debug_folder, exist_ok=True)

    def check_result(filename, reading):
        if test_cases and filename in test_cases:
            expected = str(test_cases[filename])
            if reading == expected:
                print(f"[PASS] {filename} -> {reading} (expected {expected})")
            else:
                print(f"[FAIL] {filename} -> {reading} (expected {expected})")
        else:
            print(f"[WARN] No test case for {filename}, OCR result = {reading}")

    if os.path.isfile(path):  # Case 1: Single image
        filename = os.path.basename(path)
        print(f"\nProcessing single image: {filename}")
        reading, result_image, debug_image = find_and_read_meter(path, debug_folder)
        if reading is not None:
            check_result(filename, reading)
            save_path = os.path.join(output_folder, f"result_{filename}")
            cv2.imwrite(save_path, result_image)
            print(f"Result image saved to {save_path}")
        else:
            print("Failed to read water meter from this image.")

    elif os.path.isdir(path):  # Case 2: Folder of images
        for filename in os.listdir(path):
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.join(path, filename)
                print(f"\nProcessing {filename} ...")
                reading, result_image, debug_image = find_and_read_meter(image_path, debug_folder)

                if reading is not None:
                    check_result(filename, reading)
                    save_path = os.path.join(output_folder, f"result_{filename}")
                    cv2.imwrite(save_path, result_image)
                    print(f"Result image saved to {save_path}")
                else:
                    print("Failed to read water meter from this image.")
    else:
        print("Error: Path is neither a file nor a folder.")


# --- Example usage with TEST CASES ---
expected_readings = {
    "b.png": "91234",
    "c.png": "98356",
    "d.png": "22222",
    "e.png": "11111",
    "f.png": "99988",
    "g.png": "78787",
    "h.png": "01456",
    "water_meter.png": "99988"
}
clear_images.clear_folders()
process_images("images", test_cases=expected_readings)
