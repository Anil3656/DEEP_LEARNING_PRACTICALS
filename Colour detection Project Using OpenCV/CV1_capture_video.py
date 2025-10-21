import cv2 # OpenCV library for computer vision
import numpy as np # NumPy library for numerical operations

cap = cv2.VideoCapture(0) # 0 is usually the default camera

while True:
    _, frame = cap.read()  # Capture frame-by-frame
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # Convert BGR to HSV
    
    cv2.imshow("Frame", frame) # Display the resulting frame
    
    key = cv2.waitKey(1)  # Wait for a key press
    if key == 27:  # ESC key to break
        break
    