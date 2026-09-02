"""Puntos de referencia de la cancha y filtros de zona.

Define, a partir de la calibración (4 esquinas de cancha + arcos), las zonas:
  - CANCHA: rectángulo del campo de juego.
  - AREA (área penal) por lado, aproximada a un rectángulo desde la línea de gol.
  - CORNER: esquinas del campo.

Con esto se filtran los actores que NO juegan (entrenadores, suplentes, público)
porque están fuera de la cancha, y se distinguen eventos por zona.
"""
import numpy as np


def _rect_corners_from_px(corners_px, H):
    """Proyecta 4 puntos píxel (esquinas cancha) a metros mediante homografía."""
    pts = np.array(corners_px, dtype=np.float32).reshape(-1, 1, 2)
    m = cv2_perspective(pts, H).reshape(-1, 2)
    return m


def cv2_perspective(pts, H):
    import cv2
    return cv2.perspectiveTransform(pts, H)


class PitchZones:
    """Zonas en coordenadas de metros (píxel -> homografía -> metros)."""

    def __init__(self, cal):
        self.H = cal.H
        # Esquinas de la cancha en metros (orden: TL, TR, BL, BR del calibrador)
        corners = np.array(cal.pitch_corners_px, dtype=np.float32).reshape(-1, 1, 2)
        m = cv2_perspective(corners, self.H).reshape(-1, 2)
        xs = m[:, 0]
        ys = m[:, 1]
        self.x_min, self.x_max = float(xs.min()), float(xs.max())
        self.y_min, self.y_max = float(ys.min()), float(ys.max())
        # Ancho de la cancha en metros (eje y) y largo (eje x)
        self.width = self.x_max - self.x_min
        self.length = self.y_max - self.y_min

        # Áreas de gol: arcos proyectados a metros
        gl = cv2_perspective(
            np.array(cal.goal_left, dtype=np.float32).reshape(-1, 1, 2), self.H).reshape(-1, 2)
        gr = cv2_perspective(
            np.array(cal.goal_right, dtype=np.float32).reshape(-1, 1, 2), self.H).reshape(-1, 2)
        self.goal_left_m = gl
        self.goal_right_m = gr
        # x de cada línea de gol (la coordenada del arco en el eje x)
        self.goalL_x = float(np.mean(gl[:, 0]))
        self.goalR_x = float(np.mean(gr[:, 0]))
        # rango y de cada arco (ancho del arco en metros)
        self.goalL_y = [float(gl[:, 1].min()), float(gl[:, 1].max())]
        self.goalR_y = [float(gr[:, 1].min()), float(gr[:, 1].max())]

    # ---- Filtros ----
    def inside_pitch(self, x, y, margin_m=1.0):
        """¿El punto (x,y) en metros está dentro de la cancha?"""
        return (self.x_min + margin_m <= x <= self.x_max - margin_m and
                self.y_min + margin_m <= y <= self.y_max - margin_m)

    def inside_area(self, x, y, side="L", depth_m=16.5, width_m=40.3):
        """¿Está dentro del área penal? side='L' (arco izquierdo) o 'R'."""
        if side == "L":
            x_line = self.goalL_x
            y_rng = self.goalL_y
            in_x = x_line <= x <= x_line + depth_m
        else:
            x_line = self.goalR_x
            y_rng = self.goalR_y
            in_x = x_line - depth_m <= x <= x_line
        cy = (y_rng[0] + y_rng[1]) / 2.0
        in_y = abs(y - cy) <= width_m / 2.0
        return in_x and in_y

    def corner_zone(self, x, y, radius_m=5.0):
        """Devuelve el nombre de la esquina si está en zona de córner, si no None."""
        corners = {
            "TL": (self.x_min, self.y_min),
            "TR": (self.x_max, self.y_min),
            "BL": (self.x_min, self.y_max),
            "BR": (self.x_max, self.y_max),
        }
        for name, (cx, cy) in corners.items():
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius_m ** 2:
                return name
        return None

    def pitch_rect_m(self):
        """Rectángulo [x_min, y_min, x_max, y_max] en metros."""
        return [self.x_min, self.y_min, self.x_max, self.y_max]
