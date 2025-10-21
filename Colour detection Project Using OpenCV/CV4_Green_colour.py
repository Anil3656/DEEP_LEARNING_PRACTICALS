import cv2 # OpenCV library for computer vision
import numpy as np # NumPy library for numerical operations

cap = cv2.VideoCapture(0) # 0 is usually the default camera

while True:
    _, frame = cap.read()  # Capture frame-by-frame
    
    low_red = np.array([40, 100, 100])   # Lower bound for red color in HSV
    high_red = np.array([102, 255, 255]) # Upper bound for red color in HSV

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # Convert BGR to HSV

    red_mask = cv2.inRange(hsv, low_red, high_red) # Create a mask for red color
    red = cv2.bitwise_and(frame, frame, mask=red_mask) # Apply the mask to get red regions
    
    
    cv2.imshow("Frame", frame) # Display the resulting frame
    cv2.imshow("Red Mask", red) # Display the red color mask
    
    key = cv2.waitKey(1)  # Wait for a key press
    if key == 27:  # ESC key to break
        break
    