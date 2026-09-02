"""Asignación de equipos por color de camiseta.

Para vista de dron, tomamos el color dominante del tercio superior del bbox
(la camiseta) de cada jugador y lo clasificamos en 2 equipos + árbitro usando
clustering en espacio de color HSV (k-means).
"""
import numpy as np


def _shirt_crop(frame, bbox):
    """Recorta el NÚCLEO central del jugador (evita el césped circundante).

    En vista cenital el jugador es una mancha pequeña: tomamos el núcleo
    central del bbox (40% ancho, 40% alto) donde está el cuerpo, y en el
    análisis de color EXCLUIMOS los píxeles verdes del césped.
    """
    x1, y1, x2, y2 = map(int, bbox)
    w = x2 - x1
    h = y2 - y1
    if w < 4 or h < 6:
        return None
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    x_a = max(0, cx - int(w * 0.20))
    x_b = min(frame.shape[1], cx + int(w * 0.20))
    y_a = max(0, cy - int(h * 0.20))
    y_b = min(frame.shape[0], cy + int(h * 0.20))
    if x_b - x_a < 2 or y_b - y_a < 2:
        return None
    return frame[y_a:y_b, x_a:x_b]


def _is_grass(h, s, v):
    """¿Es un píxel de césped (verde)? Se excluye del análisis de camiseta.

    Acepta arrays (operación vectorizada).
    """
    return (h >= 35) & (h <= 95) & (s > 40) & (v > 50)


def _dominant_color_hsv(crop):
    """Devuelve el color HSV dominante del crop EXCLUYENDO el césped verde.

    En vista cenital el fondo es césped; el color de la camiseta es lo que
    queda tras filtrar los píxeles verdes.
    """
    import cv2
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0].astype(float)
    s = hsv[:, :, 1].astype(float)
    v = hsv[:, :, 2].astype(float)
    # píxeles de color que NO son césped
    not_grass = ~(_is_grass(h, s, v))
    colored = not_grass & (s > 50) & (v > 50)
    if colored.sum() < 10:
        # fallback: píxeles brillantes (camisetas blancas se saturan de luz)
        bright = not_grass & (v > 120)
        if bright.sum() < 5:
            return None
        mask = bright
    else:
        mask = colored
    hh = h[mask].mean()
    ss = s[mask].mean()
    vv = v[mask].mean()
    return np.array([hh, ss, vv], dtype=np.float32)


class TeamAssigner:
    """Clasifica a los jugadores en equipo A/B/árbitro según color de camiseta.

    Dos modos:
      - 'auto': k-means sobre los colores dominantes de todos los jugadores.
      - 'manual': el usuario selecciona un crop de referencia por equipo.
    """

    def __init__(self, mode="auto", n_clusters=3):
        self.mode = mode
        self.n_clusters = n_clusters
        self.team_centers = None   # (3,3) HSV centers [teamA, teamB, ref]
        self.player_team = {}      # track_id -> team label
        self._pending_colors = []  # acumulación de colores hasta tener buena muestra

    def _fit_colors(self, colors: np.ndarray):
        """Ajusta los centros de equipo desde un array (N,3) de colores HSV."""
        from sklearn.cluster import KMeans
        k = min(self.n_clusters, len(colors))
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(colors)
        centers = km.cluster_centers_

        if k >= 3:
            # 1) árbitro: cluster con hue rojo (H<20 o H>160) y saturación alta
            ref_idx = None
            for i, c in enumerate(centers):
                h_, s_, v_ = c
                is_red = (h_ < 20) or (h_ > 160)
                if is_red and s_ > 100:
                    ref_idx = i
                    break
            if ref_idx is None:
                # fallback: cluster menos saturado = árbitro
                ref_idx = int(np.argmin(centers[:, 1]))
            rest = [i for i in range(k) if i != ref_idx]
            # 2) equipos: más saturado = color, menos = blanco/claro
            rest_sorted = sorted(rest, key=lambda i: -centers[i, 1])
            teamA, teamB = centers[rest_sorted[0]], centers[rest_sorted[1]]
            ref = centers[ref_idx]
        elif k == 2:
            teamA, teamB = centers[0], centers[1]
            ref = None
        else:
            teamA, teamB, ref = centers[0], None, None
        self.team_centers = (teamA, teamB, ref)

    def fit(self, frame, tracked_players):
        """Estima colores de equipo a partir de una snapshot de jugadores.

        Estrategia para vista cenital (fútbol amateur):
          1. Árbitro: cluster con HUE ROJO (rojo es casi universal para árbitros).
          2. Equipos: los 2 clusters restantes, ordenados por saturación
             (el más saturado = equipo de color, el menos = blanco/claro).
        """
        colors = []
        for obj in tracked_players:
            c = _shirt_crop(frame, obj.bbox)
            if c is None:
                continue
            col = _dominant_color_hsv(c)
            if col is not None:
                colors.append(col)
        if len(colors) < 6:
            return
        self._fit_colors(np.array(colors, dtype=np.float32))

    def assign(self, frame, tracked_players) -> dict[int, str]:
        """Asigna equipo a cada jugador. Devuelve {track_id: 'A'|'B'|'REF'}.

        Si aún no se ha ajustado, ACUMULA colores de varios frames hasta tener
        una muestra representativa (>= MIN_SAMPLES) antes de fitear. Esto evita
        que el frame 0 (pocos jugadores) fije un mal clustering para todo.
        """
        out = {}
        if self.team_centers is None:
            # acumular colores de este frame
            for obj in tracked_players:
                c = _shirt_crop(frame, obj.bbox)
                col = _dominant_color_hsv(c) if c is not None else None
                if col is not None:
                    self._pending_colors.append(col)
            # fitear solo con muestra suficiente y variada
            if len(self._pending_colors) >= 30:
                self._fit_colors(np.array(self._pending_colors, dtype=np.float32))
                self._pending_colors = []
            if self.team_centers is None:
                return out
        teamA, teamB, ref = self.team_centers
        for obj in tracked_players:
            c = _shirt_crop(frame, obj.bbox)
            col = _dominant_color_hsv(c) if c is not None else None
            if col is None:
                out[obj.track_id] = self.player_team.get(obj.track_id, "A")
                continue
            dists = [np.linalg.norm(col - c0) for c0 in (teamA, teamB) if c0 is not None]
            if ref is not None:
                dist_ref = np.linalg.norm(col - ref)
                if dist_ref < min(dists):
                    out[obj.track_id] = "REF"
                    self.player_team[obj.track_id] = "REF"
                    continue
            label = ["A", "B"][int(np.argmin(dists))]
            out[obj.track_id] = label
            self.player_team[obj.track_id] = label
        return out
