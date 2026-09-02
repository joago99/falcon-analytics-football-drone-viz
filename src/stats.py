"""Cálculo de estadísticas por jugador y por equipo a partir del tracking.

Acumula en el tiempo: distancia recorrida (metros), posesión, velocidad,
y resúmenes por equipo. Los eventos (gol/tiro/pase) se cuentan aparte.

Solo los tracks con frames >= MIN_TRACK_FRAMES cuentan como jugadores
(los tracks cortos son detecciones espurias / ruido del tracker).
"""
import numpy as np
from collections import defaultdict

from config import MIN_TRACK_FRAMES


class StatsAccumulator:
    def __init__(self, cal, min_frames: int = None):
        self.cal = cal
        self.min_frames = min_frames if min_frames is not None else MIN_TRACK_FRAMES
        # por track_id: {track_id: {dist, frames, speeds, team, last_pos, last_t}}
        self.players = defaultdict(lambda: {
            "dist_m": 0.0, "frames": 0, "speeds": [], "team": "A",
            "last_pos": None, "last_t": None, "samples": 0,
        })
        self.team_possession = {"A": 0.0, "B": 0.0}  # segundos de posesión

    def update(self, players, team_map, ball_m, t_sec, dt_real):
        """Acumula estadísticas para este frame.
        players: [{track_id, pos_m}] posiciones en metros."""
        if dt_real <= 0:
            dt_real = 1e-6
        # Posición de cada jugador
        for p in players:
            pid = p["track_id"]
            pr = self.players[pid]
            pr["team"] = team_map.get(pid, pr["team"])
            if pr["last_pos"] is not None:
                d = np.linalg.norm(p["pos_m"] - pr["last_pos"])
                # filtrar saltos irreales (>15m en un frame)
                if d < 15:
                    pr["dist_m"] += d
                    pr["speeds"].append(d / dt_real)
            pr["last_pos"] = p["pos_m"]
            pr["last_t"] = t_sec
            pr["frames"] += 1

        # Posesión: si hay pelota, el equipo del jugador más cercano la tiene
        if ball_m is not None and players:
            nearest = min(players, key=lambda p: np.linalg.norm(p["pos_m"] - ball_m))
            d = np.linalg.norm(nearest["pos_m"] - ball_m)
            if d < 5.0:  # pelota cerca de un jugador
                team = team_map.get(nearest["track_id"], "A")
                self.team_possession[team] += dt_real

    def summary(self):
        """Devuelve stats consolidadas por jugador y por equipo.

        Solo cuenta tracks con frames >= self.min_frames (tracks estables).
        """
        per_player = []
        for pid, pr in self.players.items():
            if pr["frames"] < self.min_frames:
                continue  # ruido / detección espuria, no es un jugador real
            speeds = pr["speeds"]
            per_player.append({
                "track_id": int(pid),
                "team": pr["team"],
                "distance_m": round(pr["dist_m"], 1),
                "frames": pr["frames"],
                "avg_speed_mps": round(float(np.mean(speeds)), 2) if speeds else 0.0,
                "max_speed_mps": round(float(np.max(speeds)), 2) if speeds else 0.0,
            })
        # ordenar por distancia
        per_player.sort(key=lambda x: -x["distance_m"])

        per_team = {}
        for team in ("A", "B"):
            tp = [p for p in per_player if p["team"] == team]
            per_team[team] = {
                "players": len(tp),
                "total_distance_m": round(sum(p["distance_m"] for p in tp), 1),
                "possession_sec": round(self.team_possession.get(team, 0.0), 1),
            }
        return {"players": per_player, "teams": per_team}
