"""Doble pase: tracker de PELOTA a alta resolución via crop+upscale.

La pelota es el objeto más difícil de detectar en vista de dron (muy pequeña):
a 640px no se detecta (0%). Estrategia:

  - Se corre el modelo YOLOv8m (640px) sobre un CROP centrado en la posición
    previa de la pelota, upscaleado 4x → resolución efectiva 4 veces mayor
    (equivalente a 1280px+ sobre la región de interés).
  - Solo se corre el crop cuando hay una posición previa (la pelota se sigue
    con el tracker de posición lineal entre samples).
  - Cada BALL_SAMPLE_EVERY frames se corre el modelo sobre todo el frame a
    640px para encontrar la pelota inicial o si se perdió totalmente.
  - Entre samples, la posición de la pelota se PROPAGA usando la velocidad
    de la última detección (lineal).
  - Si pasan BALL_STALE_FRAMES sin detección, se suelta la pelota (None).
"""
import numpy as np


class BallTracker:
    def __init__(self, model_path: str, imgsz: int = 640, conf: float = 0.20,
                 sample_every: int = 15, stale_frames: int = 30,
                 short_prop_frames: int = 45, crop_upscale: float = 4.0,
                 crop_size: int = 256):
        from ultralytics import YOLO
        import torch
        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.sample_every = max(1, sample_every)
        self.stale_frames = max(1, stale_frames)
        self.short_prop_frames = max(1, short_prop_frames)
        self.device = 0 if torch.cuda.is_available() else "cpu"
        # doble pase: crop grande alrededor de la última posición de la pelota
        self.crop_upscale = crop_upscale       # cuánto agrandar el crop en px
        self.crop_size = crop_size            # lado del crop (px)
        # estado
        self._pos_px = None        # (x, y) centro de la pelota en píxel
        self._vel_px = np.zeros(2)  # velocidad (px/frame)
        self._frames_since_det = 0
        self._frame_count = 0

    def update(self, frame) -> tuple[np.ndarray | None, str]:
        """Devuelve ((x, y) centro pelota en píxel o None, estado).

        Estado ('real' | 'prop_short' | 'prop_long' | 'none'):
          - 'real':       detección real del modelo en este frame (SIEMPRE para eventos)
          - 'prop_short': propagación ≤ SHORT_PROP_FRAMES tras detección real
                          (jugada real en curso -> también alimenta eventos)
          - 'prop_long':  propagación larga sin detección (posición sintética
                          -> solo se dibuja, NO genera eventos)
          - 'none':       sin pelota
        """
        self._frame_count += 1
        self._frames_since_det += 1

        ball_center = None

        if self._frame_count % self.sample_every == 0:
            # Sample: buscar pelota. Si tenemos posición previa, crop+upscale
            # alrededor de ella para resolución efectiva alta (doble pase).
            if self._pos_px is not None and self._frames_since_det <= self.stale_frames:
                ball_center = self._detect_in_crop(frame)
            else:
                # posición perdida o primer frame: full-frame a 640px
                ball_center = self._detect_full(frame)
        elif self._pos_px is not None and self._frames_since_det <= self.short_prop_frames:
            # frame intermedio con posición reciente: intento rápido de crop
            ball_center = self._detect_in_crop(frame)

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
        """Detección full-frame a imgsz (640px)."""
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                     verbose=False, device=self.device)
        return self._extract_ball(results)

    def _detect_in_crop(self, frame) -> np.ndarray | None:
        """Crop centrado en posición previa, upscaleado, para resolución efectiva alta.

        El crop se hace en el frame original (full res), se upscalea y se le
        pasa al modelo a imgsz. Las coordenadas resultantes se mapean de vuelta
        al frame original.
        """
        cx, cy = self._pos_px
        half = int(self.crop_size * self.crop_upscale / 2)
        h, w = frame.shape[:2]
        x1 = max(0, int(cx) - half)
        y1 = max(0, int(cy) - half)
        x2 = min(w, int(cx) + half)
        y2 = min(h, int(cy) + half)
        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        # upscale del crop para que el modelo vea detalles de la pelota
        scale = self.crop_upscale
        crop_up = __import__("cv2").resize(crop, None, fx=scale, fy=scale,
                                           interpolation=__import__("cv2").INTER_CUBIC)
        results = self.model.predict(crop_up, conf=self.conf, imgsz=self.imgsz,
                                     verbose=False, device=self.device)
        ball = self._extract_ball(results)
        if ball is not None:
            # mapeo de coordenadas del crop upscaleado -> frame original
            bx, by = ball
            # coordenadas relativas al crop_up (ya upscaleado)
            rx = (bx - x1 * scale) / scale
            ry = (by - y1 * scale) / scale
            return np.array([rx, ry], dtype=np.float32)

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
