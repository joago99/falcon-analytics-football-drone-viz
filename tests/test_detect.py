"""Quick test: corre YOLO sobre un frame real del video y muestra cuántos
jugadores y pelotas detecta. Valida el stack antes de calibrar.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2
from ultralytics import YOLO
from config import CONF_THRESHOLD, IOU_THRESHOLD, CLASS_PERSON, CLASS_SPORTS_BALL

FRAME = r"C:\Users\joaqu\football-drone-viz\calib\samples\v2_work_t5.png"

model = YOLO("yolov8m.pt")
frame = cv2.imread(FRAME)
print(f"Frame: {frame.shape[1]}x{frame.shape[0]}")

results = model.predict(frame, conf=0.25, iou=0.5, verbose=False, device=0)[0]
if results.boxes is None:
    print("Sin detecciones")
else:
    cls = results.boxes.cls.cpu().numpy().astype(int)
    names = model.names
    for c in set(cls):
        n = (cls == c).sum()
        print(f"  {names[c]}: {n} detecciones")
    print(f"TOTAL: {len(cls)}")
