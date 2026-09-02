"""Doble pase: tracker de PELOTA a alta resolución (1280px), muestreado.

La pelota es el objeto más difícil de detectar en vista de dron (muy pequeña):
a 640px no se detecta (0%), a 1280px sí (~0.44 conf). Correr el modelo de pelota a
1280px en TODOS los frames es lento. Estrategia:

  - Se corre el modelo YOLOv8m a imgsz=1280 sobre todo el frame cada
    BALL_SAMPLE_EVERY frames (full-frame high-res sample).
  - Entre samples, la posición de la pelota se PROPAGA usando la velocidad
    de la última detección (lineal).
  - Si pasan BALL_STALE_FRAMES sin detección, se suelta la pelota (None).

NOTA: el modelo yolov5m-1280 (formato YOLOv5 legacy) no es compatible con
Ultralytics YOLOv8. Se usa yolov8m-640 corrido a imgsz=1280, que SÍ detecta
la pelota en tests directos (conf ~0.44).
"""
import numpy as np


class BallTracker:
    def __init__(self, model_path: str, imgsz: int = 1280, conf: float = 0.15,
                 sample_every: int = 15, stale_frames: int = 30,
                 short_prop_frames: int = 45):
        from ultralytics import YOLO
        import torch
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.sample_every = max(1, sample_every)
        self.stale_frames = max(1, stale_frames)
        self.short_prop_frames = max(1, short_prop_frames)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        # estado
        self._pos_px = None        # (x, y) centro de la pelota en píxel
        self._vel_px = np.zeros(2)  # velocidad (px/frame)
        self._frames_since_det = 0
        self._frame_count = 0

    def update(self, frame) -> tuple[np.ndarray | None, str]:
        """Devuelve ((x, y) centro pelota en píxel o None, estado).

        Estado ('real' | 'prop_short' | 'prop_long' | 'none'):
          - 'real':       detección real del modelo en este frame (SIEMPRE para eventos)
          - 'prop_short': propagación <= SHORT_PROP_FRAMES tras detección real
                          (jugada real en curso -> también alimenta eventos)
          - 'prop_long':  propagación larga sin detección (posición sintética
                          -> solo se dibuja, NO genera eventos)
          - 'none':       sin pelota
        """
        self._frame_count += 1
        self._frames_since_det += 1

        ball_center = None
        if self._frame_count % self.sample_every == 0:
            ball_center = self._detect_full(frame)

        if ball_center is not None:
            if self._pos_px is not None:
                self._vel_px = (ball_center - self._pos_px) / max(1, self._frames_since_det)
            else:
                self._vel_px = np.zeros(2)
            self._pos_px = ball_center
            self._frames_since_det = 0
            return self._pos_px.copy(), "real"

        # no se detectó: propagar con velocidad lineal
        if self._pos_px is not None:
            self._pos_px = self._pos_px + self._vel_px
            if self._frames_since_det > self.stale_frames:
                self._pos_px = None
                self._vel_px = np.zeros(2)
                return None, "none"
            state = "prop_short" if self._frames_since_det <= self.short_prop_frames else "prop_long"
            return self._pos_px.copy(), state
        return None, "none"

    def _detect_full(self, frame) -> np.ndarray | None:
        """Detección full-frame a imgsz (1280px) para resolver pelota pequeña."""
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                     verbose=False, device=self.device)
        return self._extract_ball(results)

    def _extract_ball(self, results) -> np.ndarray | None:
        """Extrae el centro de la pelota (clase 0) de resultados de YOLO."""
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            ball_idx = np.where(cls == 0)[0]
            if len(ball_idx) > 0:
                confs = r.boxes.conf.cpu().numpy()
                bi = ball_idx[int(np.argmax(confs[ball_idx]))]
                x1, y1, x2, y2 = r.boxes.xyxy[bi].cpu().numpy()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                return np.array([cx, cy], dtype=np.float32)
        return None

    @property
    def has_ball(self) -> bool:
        return self._pos_px is not None
