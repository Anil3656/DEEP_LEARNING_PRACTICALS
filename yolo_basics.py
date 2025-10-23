from ultralytics import YOLO
import numpy as np

# Load a model
model = YOLO("yolov8n.pt", "v8")  # load a pretrained YOLOv8n model

detections = model.predict(source="https://www.shutterstock.com/image-photo/new-delhi-feb-23-traffic-260nw-1750742807.jpg", conf=0.25, save=True)  # predict on an image

print(detections)