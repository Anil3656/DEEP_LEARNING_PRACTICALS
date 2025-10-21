import cv2 # OpenCV library for computer vision
import numpy as np # NumPy library for numerical operations

cap = cv2.VideoCapture(0) # 0 is usually the default camera

while True:
    _, frame = cap.read()  # Capture frame-by-frame
    
    
    low_red = np.array([161, 155, 80])   # Lower bound for red color in HSV
    high_red = np.array([179, 255, 255]) # Upper bound for red color in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # Convert BGR to HSV
    red_mask = cv2.inRange(hsv, low_red, high_red) # Create a mask for red color
    red = cv2.bitwise_and(frame, frame, mask=red_mask)
    
    
    low_blue = np.array([94, 80, 2])   # Lower bound for red color in HSV
    high_blue = np.array([126, 255, 255]) # Upper bound for red color in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # Convert BGR to HSV
    blue_mask = cv2.inRange(hsv, low_blue, high_blue) # Create a mask for red color
    blue = cv2.bitwise_and(frame, frame, mask=blue_mask)
    
    
    low_red = np.array([40, 100, 100])   # Lower bound for red color in HSV
    high_red = np.array([102, 255, 255]) # Upper bound for red color in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # Convert BGR to HSV
    green_mask = cv2.inRange(hsv, low_red, high_red) # Create a mask for red color
    green = cv2.bitwise_and(frame, frame, mask=green_mask)
    
    
    low= np.array([0, 42, 0])   # Lower bound for red color in HSV
    high = np.array([179, 255, 255]) # Upper bound for red color in HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)   # Convert BGR to HSV
    every_mask = cv2.inRange(hsv, low, high) # Create a mask for red color
    every = cv2.bitwise_and(frame, frame, mask=every_mask)
    
    cv2.imshow("Frame", frame)
    cv2.imshow("Red", red)
    cv2.imshow("Blue", blue)
    cv2.imshow("Green", green)
    cv2.imshow("every", every)
    
    key = cv2.waitKey(1)  # Wait for a key press
    if key == 27:  # ESC key to break
        break
    