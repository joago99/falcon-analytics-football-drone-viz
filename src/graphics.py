"""Gráficas superpuestas dentro del video anotado.

  - Overlay de la cancha: líneas del campo, arcos y área penal proyectadas
    desde la calibración (se dibuja sobre cada frame).
  - Trails: estelas de los jugadores (posiciones recientes en píxel).
  - Eventos: texto en pantalla cuando ocurre un evento (GOAL/SHOT/PASS/CORNER).

Todo es puramente visual; no altera el análisis.
"""
import cv2
import numpy as np

# colores BGR
C_GREEN = (60, 200, 60)
C_RED = (60, 60, 240)
C_CYAN = (240, 220, 60)
C_GRAY = (128, 128, 128)
C_WHITE = (255, 255, 255)
C_YELLOW = (0, 255, 255)
C_BLACK = (0, 0, 0)


class VideoGraphics:
    def __init__(self, cal, zones):
        self.cal = cal
        self.zones = zones
        self.trails = {}          # track_id -> deque de (x, y) en píxel
        self.trail_len = 25
        self.recent_events = []   # (t_sec, text)

    def update(self, tracked, team_map, t_sec):
        """Actualiza las estelas de los jugadores (solo activos)."""
        for o in tracked:
            cx = float((o.bbox[0] + o.bbox[2]) / 2)
            cy = float((o.bbox[1] + o.bbox[3]) / 2)
            team = team_map.get(o.track_id, "A")
            if team in ("REF", "OUT"):
                continue
            if o.track_id not in self.trails:
                self.trails[o.track_id] = []
            tr = self.trails[o.track_id]
            tr.append((cx, cy))
            if len(tr) > self.trail_len:
                tr.pop(0)

    def draw_pitch(self, frame):
        """Dibuja la cancha (líneas + arcos) proyectada en píxel."""
        H_inv = np.linalg.inv(self.cal.H)

        def m2px(mx, my):
            p = np.array([mx, my], dtype=np.float32).reshape(-1, 1, 2)
            out = cv2.perspectiveTransform(p, H_inv).reshape(-1, 2)[0]
            return int(out[0]), int(out[1])

        # borde de la cancha (105x68)
        pts = [m2px(0, 0), m2px(105, 0), m2px(105, 68), m2px(0, 68)]
        cv2.polylines(frame, [np.array(pts, np.int32)], True, C_WHITE, 2)

        # línea media
        cv2.line(frame, m2px(52.5, 0), m2px(52.5, 68), C_WHITE, 2)
        # círculo central
        cv2.circle(frame, m2px(52.5, 34), int(9.15 / 105 * (m2px(105, 0)[0] - m2px(0, 0)[0])),
                   C_WHITE, 2)

        # arcos
        for gl, glx in ((self.cal.goal_left, 0), (self.cal.goal_right, 105)):
            a = m2px(glx, 34 - 3.66)
            b = m2px(glx, 34 + 3.66)
            cv2.line(frame, a, b, C_YELLOW, 4)

    def draw_trails(self, frame):
        """Dibuja estelas de movimiento de los jugadores activos."""
        for tid, tr in self.trails.items():
            if len(tr) < 2:
                continue
            pts = np.array(tr, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, C_CYAN, 2)

    def push_event(self, text, t_sec, priority: int = 9):
        """Registra un evento en pantalla. Los de mayor prioridad (menor número)
        reemplazan a los de menor prioridad de la misma ventana temporal."""
        self.recent_events.append((t_sec, text, priority))
        # si el último evento es más importante que el anterior, reemplazarlo
        if len(self.recent_events) >= 2:
            last, prev = self.recent_events[-1], self.recent_events[-2]
            if abs(last[0] - prev[0]) < 3.0 and last[2] < prev[2]:
                self.recent_events[-2] = last
                self.recent_events.pop()
        if len(self.recent_events) > 5:
            self.recent_events.pop(0)

    def draw_events(self, frame, t_sec):
        """Dibuja texto de eventos recientes (dentro de 3s) en pantalla."""
        y = 60
        for ev_t, text, _prio in self.recent_events:
            if t_sec - ev_t > 3.0:
                continue
            cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, C_YELLOW, 3)
            y += 45

    def draw(self, frame, tracked, team_map, t_sec, ball_px=None):
        # Sin overlay de cancha ni trails: solo jugadores/árbitros/balón
        self._draw_players(frame, tracked, team_map)
        self._draw_ball(frame, ball_px)
        self.draw_events(frame, t_sec)

    def _draw_ball(self, frame, ball_px):
        """Dibuja la pelota detectada (círculo amarillo + cruz)."""
        if ball_px is None:
            return
        x, y = int(ball_px[0]), int(ball_px[1])
        cv2.circle(frame, (x, y), 12, (0, 255, 255), 3)
        cv2.line(frame, (x - 18, y), (x + 18, y), (0, 255, 255), 1)
        cv2.line(frame, (x, y - 18), (x, y + 18), (0, 255, 255), 1)

    def _draw_players(self, frame, tracked, team_map):
        """Dibuja a los jugadores con círculos estilo FIFA 2025 (sin bboxes).

        Cada detección se enmarca en un círculo relleno del color de su equipo,
        con borde blanco y el track ID centrado dentro.
        Colores: A=verde, B=rojo, REF=cian (árbitro), OUT=gris (staff),
        UNK=magenta (sin equipo confiable).
        """
        for o in tracked:
            x1, y1, x2, y2 = map(int, o.bbox)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            team = team_map.get(o.track_id, "?")
            color = {"A": (60, 200, 60), "B": (60, 60, 240),
                     "REF": (0, 220, 220), "OUT": (160, 160, 160),
                     "UNK": (255, 0, 255)}.get(team, (255, 255, 255))
            # radio proporcional al tamaño del jugador (vista cenital)
            r = int(max(x2 - x1, y2 - y1) / 2) + 6
            # círculo relleno + borde blanco
            cv2.circle(frame, (cx, cy), r, color, -1)
            cv2.circle(frame, (cx, cy), r, (255, 255, 255), 2)
            # ID centrado dentro del círculo (negro con contorno blanco)
            label = str(o.track_id)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx, ty = cx - tw // 2, cy + th // 2
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
