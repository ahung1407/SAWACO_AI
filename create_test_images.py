
import cv2
import numpy as np
import os

def create_digit_image(digit, filename):
    # Create a black image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    color = (255, 255, 255) # White
    thickness = 5
    
    # Get text size to center it
    text_size = cv2.getTextSize(str(digit), font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    
    # Put text (digit) on image
    cv2.putText(img, str(digit), (text_x, text_y), font, font_scale, color, thickness)
    
    # Save
    cv2.imwrite(filename, img)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_digit_image(3, "test_digit_3.jpg")
    create_digit_image(5, "test_digit_5.jpg")
    create_digit_image(9, "test_digit_9.jpg")
    create_digit_image(0, "test_digit_0.jpg")
