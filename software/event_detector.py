from ultralytics import YOLO
import cv2          
import numpy as np  

MODEL_PATH = "../model/best_v7_32.pt"
model = YOLO(MODEL_PATH)

def detect_events(image, conf=0.15, iou=0.5):
    results = model.predict(image, conf=conf, iou=iou, verbose=False)
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            'center_x': (x1 + x2) / 2,
            'center_y': (y1 + y2) / 2,
            'width': x2 - x1,
            'height': y2 - y1,
            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
            'confidence': float(box.conf[0])
        })
    # Hoechste Konfidenz zuerst - direkt die Reihenfolge fuer die Queue
    detections.sort(key=lambda d: d['confidence'], reverse=True)
    return detections
