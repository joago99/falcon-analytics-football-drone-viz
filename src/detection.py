"""Detección de jugadores y pelota con YOLO (ultralytics)."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray      # [x1, y1, x2, y2] en píxeles
    class_id: int
    confidence: float
    track_id: int | None = None  # asignado por el tracker


_DEVICE = None

def detect(model, frame: np.ndarray, conf: float, iou: float, classes=None, imgsz=None):
    """Corre YOLO y devuelve lista de Detection (sin tracking).

    imgsz: tamaño de inferencia (menor = más rápido, menos preciso).
    """
    global _DEVICE
    if _DEVICE is None:
        import torch
        _DEVICE = 0 if torch.cuda.is_available() else "cpu"
    kw = dict(conf=conf, iou=iou, classes=classes, verbose=False, device=_DEVICE)
    if imgsz:
        kw["imgsz"] = imgsz
    results = model.predict(frame, **kw)
    dets = []
    for r in results:
        if r.boxes is None:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        for x, c, s in zip(xyxy, cls, confs):
            dets.append(Detection(bbox=np.array(x, dtype=np.float32),
                                  class_id=int(c), confidence=float(s)))
    return dets
