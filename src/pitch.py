"""Homografía: convierte coordenadas de píxel del video a metros reales de la cancha."""
from dataclasses import dataclass

import numpy as np

from config import CALIB_FILE


@dataclass
class Calibration:
    H: np.ndarray            # 3x3 homografía píxel -> metro
    goal_left: np.ndarray    # 2x2 píxeles (postes) arco izquierdo
    goal_right: np.ndarray   # 2x2 píxeles (postes) arco derecho
    pitch_length: float
    pitch_width: float
    pitch_corners_px: np.ndarray = None  # 4 esquinas de la cancha en píxel

    def load():
        import json
        if not CALIB_FILE.exists():
            raise FileNotFoundError(
                f"No hay calibración en {CALIB_FILE}. "
                "Ejecuta primero: python run_calibrate.py videos/partido.mp4"
            )
        with open(CALIB_FILE) as f:
            d = json.load(f)
        corners = d.get("corners_ordered_px", d.get("corners_px"))
        return Calibration(
            H=np.array(d["homography"], dtype=np.float32),
            goal_left=np.array(d["goal_left_px"], dtype=np.float32),
            goal_right=np.array(d["goal_right_px"], dtype=np.float32),
            pitch_length=d["pitch_length"],
            pitch_width=d["pitch_width"],
            pitch_corners_px=np.array(corners, dtype=np.float32) if corners else None,
        )

    def to_pitch(self, points_px: np.ndarray) -> np.ndarray:
        """Convierte puntos (N,2) píxel -> (N,2) metros."""
        pts = np.asarray(points_px, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2_perspective_transform(pts, self.H)
        return out.reshape(-1, 2)


def cv2_perspective_transform(pts, H):
    import cv2
    return cv2.perspectiveTransform(pts, H)


def goal_lines_meters(cal: Calibration) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve las líneas de gol (2 puntos cada una) en metros (y,z)."""
    gl = cal.to_pitch(cal.goal_left)
    gr = cal.to_pitch(cal.goal_right)
    return gl, gr
