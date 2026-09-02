"""Tracking de jugadores y pelota.

Usa el tracker NATIVO de Ultralytics (BoT-SORT por defecto), que:
  - Mantiene IDs estables ante oclusiones (apariencia + movimiento).
  - Es la opción recomendada para escenas multitudinarias (vs ByteTrack puro).
  - Reduce drásticamente los IDs espurios que inflan el conteo de jugadores.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class TrackedObject:
    track_id: int
    bbox: np.ndarray      # [x1, y1, x2, y2]
    class_id: int
    confidence: float
    centroid_px: np.ndarray  # (x, y) centro del bbox en píxel


class Tracker:
    def __init__(self, model, conf: float, iou: float, imgsz=None,
                 tracker_name: str = "botsort.yaml"):
        self.model = model
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.tracker_name = tracker_name  # "bytetrack.yaml" | "botsort.yaml"
        self._device = None

    def update(self, frame) -> list[TrackedObject]:
        """Corre YOLO + tracker sobre el frame y devuelve TrackedObject.

        Usa model.track() con persist=True para mantener IDs entre frames.
        """
        if self._device is None:
            import torch
            self._device = 0 if torch.cuda.is_available() else "cpu"
        kw = dict(conf=self.conf, iou=self.iou, verbose=False, device=self._device,
                  persist=True, tracker=self.tracker_name)
        if self.imgsz:
            kw["imgsz"] = self.imgsz
        results = self.model.track(frame, **kw)
        result = results[0]
        if result.boxes is None or result.boxes.id is None:
            return []
        xyxy = result.boxes.xyxy.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)

        out = []
        for i in range(len(xyxy)):
            box = xyxy[i].astype(np.float32)
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            out.append(TrackedObject(
                track_id=int(ids[i]),
                bbox=box,
                class_id=int(cls[i]),
                confidence=float(confs[i]),
                centroid_px=np.array([cx, cy], dtype=np.float32),
            ))
        return out
