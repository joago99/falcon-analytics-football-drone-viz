"""Detección de eventos de fútbol a partir del tracking y la homografía.

Eventos que se detectan (heurísticas sobre la posición de la pelota en metros):
  - GOAL: la pelota cruza la línea de gol (dentro del ancho del arco).
  - SHOT: la pelota se mueve rápido hacia un arco.
  - PASS: cambio de posesión, pelota recorre >= PASS_DISTANCE_M entre jugadores.
  - BALL_OUT: la pelota sale del perímetro de la cancha.
  - CORNER/CROSS: (opcional) pelota en zona de esquina tras fuera.
"""
import numpy as np

from config import GOAL_TOLERANCE_M, PASS_DISTANCE_M, SHOT_SPEED_MPS, PITCH_LENGTH, PITCH_WIDTH
from src.zones import PitchZones


class EventDetector:
    def __init__(self, cal: "Calibration"):
        self.cal = cal
        self.zones = PitchZones(cal)
        gl, gr = self._goal_lines()
        # Representar cada línea de gol como un segmento [p0, p1] en metros (x,y)
        self.goal_left = gl
        self.goal_right = gr
        # Líneas de gol verticales: x constante = posición del arco
        self.goalL_x = float(np.mean([gl[0][0], gl[1][0]]))
        self.goalR_x = float(np.mean([gr[0][0], gr[1][0]]))
        self.goal_y_range_L = [min(gl[0][1], gl[1][1]), max(gl[0][1], gl[1][1])]
        self.goal_y_range_R = [min(gr[0][1], gr[1][1]), max(gr[0][1], gr[1][1])]
        # Estados previos
        self._prev_ball_pos = None
        self._prev_time = None
        self._possession_team = None
        self._last_possessor = None
        self._events = []

    def _goal_lines(self):
        import cv2
        gl_px = self.cal.goal_left.astype(np.float32).reshape(-1, 1, 2)
        gr_px = self.cal.goal_right.astype(np.float32).reshape(-1, 1, 2)
        gl = cv2.perspectiveTransform(gl_px, self.cal.H).reshape(-1, 2)
        gr = cv2.perspectiveTransform(gr_px, self.cal.H).reshape(-1, 2)
        return gl, gr

    def _dist_to_line_x(self, ball_x, line_x):
        return abs(ball_x - line_x)

    def step(self, ball_m: np.ndarray | None, players: list, team_map: dict, t_sec: float):
        """Procesa un frame. ball_m: (x,y) en metros de la pelota o None.
        players: lista de {track_id, pos_m} (posición en metros)."""
        # --- SHOT / GOAL por velocidad y cruce de línea ---
        if ball_m is not None and self._prev_ball_pos is not None and self._prev_time is not None:
            dt = t_sec - self._prev_time
            if dt > 1e-6:
                dist = np.linalg.norm(ball_m - self._prev_ball_pos)
                speed = dist / dt
                # Tiro: alta velocidad hacia la línea de gol
                if speed > SHOT_SPEED_MPS:
                    nearL = self._dist_to_line_x(ball_m[0], self.goalL_x) < 15
                    nearR = self._dist_to_line_x(ball_m[0], self.goalR_x) < 15
                    if nearL or nearR:
                        self._events.append({"type": "SHOT", "time": t_sec,
                                             "ball": ball_m.tolist(), "speed": round(float(speed), 1)})
                # Gol: cruzó la línea de gol dentro del ancho del arco
                if self._crossed_goal(ball_m, self._prev_ball_pos):
                    self._events.append({"type": "GOAL", "time": t_sec,
                                         "ball": ball_m.tolist()})

        # --- PASS por distancia recorrida con cambio de posesión ---
        if ball_m is not None and players:
            pass_ev = self._detect_pass(ball_m, players, team_map, t_sec)
            if pass_ev:
                self._events.append(pass_ev)

        # --- BALL_OUT: pelota fuera de la cancha ---
        if ball_m is not None:
            if (ball_m[0] < -1 or ball_m[0] > PITCH_LENGTH + 1 or
                    ball_m[1] < -1 or ball_m[1] > PITCH_WIDTH + 1):
                self._events.append({"type": "BALL_OUT", "time": t_sec,
                                     "ball": ball_m.tolist()})

        # --- CORNER: balón en zona de esquina (dentro del campo) ---
        if ball_m is not None:
            corner = self.zones.corner_zone(ball_m[0], ball_m[1], radius_m=6.0)
            if corner is not None:
                self._events.append({"type": "CORNER", "time": t_sec,
                                     "corner": corner, "ball": ball_m.tolist()})

        self._prev_ball_pos = ball_m
        self._prev_time = t_sec
        return list(self._events)

    def _crossed_goal(self, ball, prev_ball):
        """La pelota cruzó una línea de gol dentro del ancho del arco."""
        # Para el arco izquierdo: la pelota pasó de x>line a x<line (o viceversa para derecho)
        for line_x, yrange, side in [
            (self.goalL_x, self.goal_y_range_L, "L"),
            (self.goalR_x, self.goal_y_range_R, "R"),
        ]:
            before = prev_ball[0] - line_x
            after = ball[0] - line_x
            crossed = (before > 0) != (after > 0)  # cambió de lado
            if crossed and abs(prev_ball[0] - line_x) < 8 and abs(ball[0] - line_x) < 8:
                # dentro del ancho del arco (eje y)
                if yrange[0] - GOAL_TOLERANCE_M <= ball[1] <= yrange[1] + GOAL_TOLERANCE_M:
                    return True
        return False

    def _detect_pass(self, ball_m, players, team_map, t_sec):
        """Pase: pelota pasó de un jugador de un equipo a otro del mismo equipo,
        recorriendo >= PASS_DISTANCE_M."""
        # find nearest player
        nearest = None
        best = 1e9
        for p in players:
            d = np.linalg.norm(p["pos_m"] - ball_m)
            if d < best:
                best = d
                nearest = p
        if nearest is None or best > 4.0:  # pelota lejos de cualquier jugador
            return None
        team = team_map.get(nearest["track_id"], "A")
        if self._possession_team is not None and team == self._possession_team:
            if self._prev_ball_pos is not None:
                travelled = np.linalg.norm(ball_m - self._prev_ball_pos)
                # Cambio de jugador manteniendo equipo, distancia suficiente
                if travelled > PASS_DISTANCE_M and nearest["track_id"] != self._last_possessor:
                    self._last_possessor = nearest["track_id"]
                    return {"type": "PASS", "time": t_sec, "team": team,
                            "ball": ball_m.tolist()}
        else:
            # Cambio de posesión
            self._possession_team = team
            self._last_possessor = nearest["track_id"]
        return None

    def get_events(self):
        return self._events
