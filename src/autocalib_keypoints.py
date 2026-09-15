"""Autocalibración de la homografía mediante keypoints de campo (Roboflow).

Mejora documentada en docs/analisis-tactico-cv.md: en lugar de la
calibración manual de 8 puntos, un modelo YOLOv8-pose detecta 32 marcas
reglamentarias del campo (cruces de líneas, penales, círculo central) y
RANSAC estima la homografía dinámica por frame.

REQUIERE API key de Roboflow (modelo serverless football-field-detection-f07vi).
Sin key, se usa la calibración manual existente (calib/calib.json).

Modelo: https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi
"""
import os

import numpy as np

ROBOFLOW_MODEL_ID = "football-field-detection-f07vi/17"


class KeypointAutocalibration:
    """Estima H píxel->metro por frame con keypoints de campo + RANSAC."""

    # Coordenadas canónicas de las marcas reglamentarias (FIFA 105x68)
    # según el dataset football-field-detection (índice -> metros).
    CANONICAL = {
        0: (0.0, 0.0),       # córner superior izquierdo
        1: (0.0, 68.0),      # córner inferior izquierdo
        2: (105.0, 0.0),     # córner superior derecho
        3: (105.0, 68.0),    # córner inferior derecho
        4: (52.5, 34.0),     # centro del campo
    }

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        self.model = None
        if self.api_key:
            try:
                from inference_sdk import InferenceHTTPClient, InferenceConfiguration
                self.client = (InferenceHTTPClient(
                    api_url="https://serverless.roboflow.com",
                    api_key=self.api_key).configure(InferenceConfiguration(
                        api_key_transport="header")))
            except ImportError:
                print("inference-sdk no instalado. pip install inference-sdk")
                self.client = None
        else:
            self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def estimate_H(self, frame) -> np.ndarray | None:
        """Devuelve homografía 3x3 píxel->metro o None si no hay suficientes keypoints."""
        if not self.available:
            return None
        try:
            result = self.client.infer(frame, model_id=ROBOFLOW_MODEL_ID)
            keypoints = result.get("predictions", [])
        except Exception as e:
            print(f"[keypoints] inferencia falló: {e}")
            return None

        src, dst = [], []
        for kp in keypoints:
            idx = kp.get("class_id") or kp.get("class")
            if idx not in self.CANONICAL:
                continue
            conf = kp.get("confidence", 0.0)
            if conf < 0.5:
                continue
            src.append([kp["x"], kp["y"]])
            dst.append(self.CANONICAL[idx])
        if len(src) < 4:
            return None
        H, _ = cv2_find_homography(np.array(src, np.float32),
                                   np.array(dst, np.float32))
        return H


def cv2_find_homography(src, dst):
    import cv2
    return cv2.findHomography(src, dst, cv2.RANSAC, 5.0)[0]
