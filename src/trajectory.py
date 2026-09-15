"""Registro de trayectorias métricas y exportación CSV (gemelo digital táctico).

Fase 3 del pipeline documentado en docs/analisis-tactico-cv.md:
las posiciones métricas de cada jugador se exportan en formato tabular
(frame, id, x_m, y_m) para ser leídas por Blender (bpy), Tableau, o
cualquier analítica táctica posterior.

Formato normalizado [0,1] opcional (estilo Wunderscout): las coordenadas
se dividen por pitch_length/pitch_width para ser independientes de la
cancha real.
"""
import csv
from pathlib import Path


class TrajectoryRecorder:
    """Acumula posiciones métricas por frame y exporta CSV."""

    def __init__(self, pitch_length: float = 105.0, pitch_width: float = 68.0,
                 normalized: bool = False):
        self.rows = []          # (frame, track_id, team, x_m, y_m)
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.normalized = normalized

    def record(self, frame_idx: int, track_id, team: str, pos_m):
        """Registra una posición métrica (pos_m = [x, y] en metros)."""
        x, y = float(pos_m[0]), float(pos_m[1])
        if self.normalized:
            x = x / self.pitch_length
            y = y / self.pitch_width
        self.rows.append((frame_idx, int(track_id), team, round(x, 3), round(y, 3)))

    def export(self, path: str | Path) -> str:
        """Escribe el CSV y devuelve la ruta."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame", "id_jugador", "equipo", "campo_x_metros", "campo_y_metros"])
            w.writerows(self.rows)
        return str(path)

    @property
    def size(self) -> int:
        return len(self.rows)
