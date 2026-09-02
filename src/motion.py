"""Compensación de movimiento de cámara (pan/tilt del dron).

El dron PANEA de lado a lado, por lo que la homografía estática de
calibración queda obsoleta a los pocos frames. Solución 100% local:

  - Cada frame se alinea con el anterior usando ECC (Enhanced Correlation
    Coefficient, cv2.findTransformECC) con modelo de homografía.
  - El warp W_t (píxel en frame t -> píxel en frame 0) se compone con la
    homografía base:  metros = H_base(W_t(pixel_t)).

  H_dyn = H_base @ W_t

Esto convierte la cámara móvil en una cámara virtual fija (stabilized)
sin depender de la nube ni de keypoints de cancha por frame.
"""
import cv2
import numpy as np


class MotionCompensator:
    def __init__(self, H_base: np.ndarray, use_ecc: bool = True,
                 max_iter: int = 60, eps: float = 1e-4):
        self.H_base = np.asarray(H_base, dtype=np.float64)
        self.use_ecc = use_ecc
        self.max_iter = max_iter
        self.eps = eps
        self._prev_gray = None
        self._W = np.eye(3, dtype=np.float64)   # warp acumulado frame->frame0

    def reset(self):
        self._prev_gray = None
        self._W = np.eye(3, dtype=np.float64)

    def update(self, frame) -> np.ndarray:
        """Devuelve la homografía dinámica H_dyn (píxel t -> metros) para este frame."""
        if not self.use_ecc:
            return self.H_base

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return self.H_base

        # ECC entre frame anterior (template) y actual (moving)
        warp = np.eye(3, dtype=np.float64)
        try:
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.max_iter, self.eps)
            _, warp = cv2.findTransformECC(
                self._prev_gray, gray, warp,
                cv2.MOTION_HOMOGRAPHY, criteria,
                None, 5,  # gaussFiltSize pequeño = más rápido
            )
        except cv2.error:
            # ECC falla si el movimiento es muy grande; mantener warp identidad
            pass

        # componer warp acumulado: W_t = W_prev @ warp_inv?  Cuidado con la dirección.
        # ECC devuelve warp tal que template ≈ warp(moving). Para alinear frame t
        # al frame 0 necesitamos: p0 = W_{t-1} @ warp @ p_t
        # (moving -> template es warp; template ya está en el sistema de frame 0 vía W_prev)
        self._W = self._W @ warp
        self._prev_gray = gray

        # homografía dinámica: metros = H_base(W_t(p))
        H_dyn = self.H_base @ self._W
        return H_dyn.astype(np.float32)
