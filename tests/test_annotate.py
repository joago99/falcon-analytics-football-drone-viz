"""Genera una imagen anotada: corre YOLO sobre el frame y dibuja bboxes.
Produce un PNG que puedes abrir para verificar la detección visualmente.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO

FRAME = r"C:\Users\joaqu\football-drone-viz\calib\samples\v2_work_t5.png"
OUT = r"C:\Users\joaqu\football-drone-viz\outputs\detect_demo.png"

model = YOLO("yolov8m.pt")
frame = cv2.imread(FRAME)
results = model.predict(frame, conf=0.25, iou=0.5, verbose=False, device=0)[0]

names = model.names
for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    cls = int(box.cls[0])
    conf = float(box.conf[0])
    color = (0, 255, 0) if names[cls] == "person" else (255, 0, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{names[cls]} {conf:.2f}", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

cv2.imwrite(OUT, frame)
print(f"Guardado en {OUT}")
print(f"Detectados: person={sum(1 for b in results.boxes if int(b.cls[0])==0)}")
