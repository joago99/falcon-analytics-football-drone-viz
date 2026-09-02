"""Orquestador principal: video -> detección/tracking -> equipos -> eventos ->
estadísticas -> highlights -> informe.

Uso típico:
  python run_pipeline.py videos/partido.mp4
Requiere una calibración previa (python run_calibrate.py videos/partido.mp4).
"""
import argparse
import json
import os
import sys
from pathlib import Path

# El venv del proyecto es la fuente de verdad: limpiar PYTHONPATH global
# (que apunta al venv de Hermes y hace shadowing de transformers/PIL/etc).
# sys.path ya se construyó al arrancar; filtramos las entradas de Hermes.
_HERMES_MARK = "hermes-agent"
sys.path = [p for p in sys.path if _HERMES_MARK not in p]
if "PYTHONPATH" in os.environ:
    os.environ.pop("PYTHONPATH")

import cv2
import numpy as np

from config import (CONF_THRESHOLD, IOU_THRESHOLD,
                    CLASS_BALL, CLASS_PLAYER, CLASS_GOALKEEPER, CLASS_REFEREE,
                    VIDEOS_DIR, OUTPUTS_DIR, HIGHLIGHTS_DIR, CALIB_FILE, YOLO_MODEL,
                    BALL_MODEL, BALL_IMGSZ, BALL_CONF, BALL_SAMPLE_EVERY, BALL_STALE_FRAMES,
                    BALL_SHORT_PROP_FRAMES)
from src.calibrate import calibrate_video
from src.pitch import Calibration
from src.tracking import Tracker
from src.ball_tracker import BallTracker
from src.teams import TeamAssigner
from src.teams_siglip import SiglipTeamClassifier
from src.events import EventDetector
from src.stats import StatsAccumulator
from src.highlights import generate_highlights
from src.zones import PitchZones
from src.graphics import VideoGraphics
from src.motion import MotionCompensator


def resolve_input(input_arg: str) -> str:
    p = Path(input_arg)
    if p.exists():
        return str(p)
    cand = VIDEOS_DIR / input_arg
    if cand.exists():
        return str(cand)
    sys.exit(f"No se encontró el video: {input_arg} (probado también en {VIDEOS_DIR})")


def ensure_calibration(video_path: str):
    if not CALIB_FILE.exists():
        print("No hay calibración para este video. Lanzando calibración interactiva...")
        calibrate_video(video_path)
    return Calibration.load()


def _json_default(o):
    """Convierte numpy types a tipos nativos para json.dump."""
    import numpy as np
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def main():
    ap = argparse.ArgumentParser(description="Pipeline de analítica de fútbol por dron")
    ap.add_argument("video", help="Ruta al video .mp4")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="Usar calibración existente sin recalibrar")
    ap.add_argument("--model", default=YOLO_MODEL, help="Modelo YOLO (pesos de fútbol por defecto)")
    ap.add_argument("--sampling", type=int, default=1,
                    help="Procesar 1 de cada N frames (1=todos)")
    ap.add_argument("--no-highlights", action="store_true",
                    help="No generar clips de highlights")
    ap.add_argument("--limit", type=int, default=0,
                    help="Procesar solo los primeros N segundos (debug)")
    ap.add_argument("--imgsz", type=int, default=0,
                    help="Tamaño de inferencia YOLO (menor=mas rapido, ej. 640)")
    ap.add_argument("--no-graphics", action="store_true",
                    help="No dibujar overlays de cancha/trails en el video")
    ap.add_argument("--teams", choices=["siglip", "hsv"], default="siglip",
                    help="Clasificador de equipos: siglip (embeddings, robusto) o hsv (color)")
    args = ap.parse_args()

    video_path = resolve_input(args.video)
    cal = ensure_calibration(video_path)

    print("Cargando modelo YOLO...")
    from ultralytics import YOLO
    model = YOLO(args.model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}x{height} @ {fps:.1f}fps, {total} frames")

    tracker = Tracker(model, CONF_THRESHOLD, IOU_THRESHOLD, imgsz=args.imgsz or None)
    # Compensación de movimiento de cámara (dron panea): homografía dinámica
    print("Inicializando compensación de movimiento de cámara (ECC)...")
    motion_comp = MotionCompensator(cal.H, use_ecc=True)
    # Doble pase: pelota a 1280px muestreada (jugadores ya van a 640 en 'tracker')
    print(f"Cargando modelo de pelota {BALL_MODEL} (imgsz={BALL_IMGSZ})...")
    ball_tracker = BallTracker(BALL_MODEL, imgsz=BALL_IMGSZ, conf=BALL_CONF,
                               sample_every=BALL_SAMPLE_EVERY, stale_frames=BALL_STALE_FRAMES,
                               short_prop_frames=BALL_SHORT_PROP_FRAMES)
    # Clasificador de equipos: SigLIP (embeddings) o HSV (color)
    if args.teams == "siglip":
        print("Cargando clasificador de equipos SigLIP (embeddings visuales)...")
        teams = SiglipTeamClassifier()
    else:
        teams = TeamAssigner()
    events_det = EventDetector(cal)
    stats = StatsAccumulator(cal)
    zones = PitchZones(cal)
    graphics = VideoGraphics(cal, zones) if not args.no_graphics else None

    # ---- Numeración correlativa: cada video anotado lleva un sufijo secuencial
    # por día: annotated_vYYYYMMDD-v001.mp4. Incrementa si el archivo existe.
    from datetime import date
    today_tag = date.today().strftime("%Y%m%d")
    stem_vid = Path(video_path).stem
    def _annotated_path(stem_vid: str) -> str:
        n = 1
        while True:
            cand = OUTPUTS_DIR / f"{stem_vid}_annotated_v{today_tag}-v{n:03d}.mp4"
            if not cand.exists():
                return str(cand)
            n += 1
    out_video = None
    if not args.no_highlights:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = _annotated_path(stem_vid)
        out_video = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_idx = 0
    max_frames = int(fps * args.limit) if args.limit > 0 else total
    prev_t = None
    # filtro de árbitros: track_id -> frames como REF (consistencia temporal)
    ref_track_frames: dict = {}
    MAX_OFFICIALS = 3   # 1 árbitro central + 2 asistentes (líneas)

    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= max_frames:
            break
        t_sec = frame_idx / fps
        dt_real = (t_sec - prev_t) if prev_t is not None else 1.0 / fps
        prev_t = t_sec

        if frame_idx % args.sampling == 0:
            # homografía dinámica para este frame (compensa pan del dron)
            H_dyn = motion_comp.update(frame)
            # helper de conversión píxel -> metros con la H dinámica
            def to_meters(px):
                import cv2 as _cv2
                pts = np.asarray(px, dtype=np.float32).reshape(-1, 1, 2)
                return _cv2.perspectiveTransform(pts, H_dyn).reshape(-1, 2)[0]

            tracked = tracker.update(frame)
            players = [o for o in tracked if o.class_id in (CLASS_PLAYER, CLASS_GOALKEEPER)]
            referees = [o for o in tracked if o.class_id == CLASS_REFEREE]
            balls = [o for o in tracked if o.class_id == CLASS_BALL]

            # --- Filtro de árbitros: consistencia temporal + máximo de oficiales.
            # El modelo de fútbol a veces marca jugadores como referee. Solo
            # los tracks que aparecen como REF en suficientes frames y están
            # dentro del top-3 de oficiales cuentan; el resto se reclasifica
            # como jugador normal.
            for r in referees:
                ref_track_frames[r.track_id] = ref_track_frames.get(r.track_id, 0) + 1
            top_refs = sorted(ref_track_frames, key=lambda k: -ref_track_frames[k])[:MAX_OFFICIALS]
            confirmed_refs = [r for r in referees
                              if r.track_id in top_refs and ref_track_frames[r.track_id] >= 20]
            # los REF no confirmados vuelven a ser jugadores
            fake_refs = [r for r in referees if r not in confirmed_refs]
            players = players + fake_refs

            # árbitro: clase propia del modelo de fútbol -> siempre REF (excluido)
            team_map = {o.track_id: "REF" for o in confirmed_refs}

            # equipos: solo para jugadores (no árbitros)
            if players:
                if isinstance(teams, SiglipTeamClassifier):
                    player_teams = teams.update(frame, players, frame_idx=frame_idx)
                else:
                    player_teams = teams.assign(frame, players)
                team_map.update(player_teams)

            # posiciones en metros + filtro de zona (excluye entrenadores/suplentes)
            players_m = []
            for o in players:
                pos_m = to_meters(o.centroid_px)
                # Solo jugadores DENTRO de la cancha cuentan (fuera = staff/suplentes)
                if not zones.inside_pitch(pos_m[0], pos_m[1], margin_m=1.5):
                    team_map[o.track_id] = "OUT"
                    continue
                players_m.append({"track_id": o.track_id, "pos_m": pos_m})

            # excluir árbitro de jugadores activos (pero se conserva en team_map
            # para el video anotado con etiqueta REF); con SigLIP, los tracks
            # sin asignación confiable (UNK) tampoco cuentan en stats/eventos
            players_active = []
            for p in players_m:
                team = team_map.get(p["track_id"], "A")
                if team in ("REF", "UNK"):
                    if team == "UNK":
                        team_map[p["track_id"]] = "UNK"
                    continue
                players_active.append(p)

            # PELOTA: doble pase — alta resolución muestreada (BallTracker).
            # Estado: 'real' (detección) o 'prop_short' (propagación corta tras
            # detección real) alimentan eventos; 'prop_long' (sintética) solo
            # se dibuja. La propagada SIEMPRE se dibuja para ver el flujo.
            ball_px, ball_state = ball_tracker.update(frame)
            ball_m = None
            if ball_px is not None:
                ball_m = to_meters(ball_px)
            # eventos solo con pelota real o propagación corta (jugada en curso)
            ball_for_events = ball_m if ball_state in ("real", "prop_short") else None

            # eventos: solo jugadores activos (sin árbitro, sin staff) y
            # solo con pelota de detección REAL (no propagada)
            before = len(events_det.get_events())
            events_det.step(ball_for_events, players_active, team_map, t_sec)
            new_events = events_det.get_events()[before:]
            if graphics is not None:
                for ev in new_events:
                    label = {"GOAL": "GOOOL!", "SHOT": "TIRO",
                             "PASS": "PASE", "CORNER": "CORNELIO",
                             "BALL_OUT": "FUERA"}.get(ev["type"], ev["type"])
                    graphics.push_event(f"{label} @ {t_sec:.0f}s", t_sec)

            # stats: igual, sin REF/OUT; posesión con pelota real/prop_short
            stats.update(players_active, team_map, ball_for_events, t_sec, dt_real)

            # anotación (muestra TODOS: jugadores, REF, OUT y la pelota)
            if out_video:
                if graphics is not None:
                    graphics.draw(frame, tracked, team_map, t_sec, ball_px=ball_px)
                else:
                    _annotate(frame, tracked, team_map, ball_m)

        frame_idx += 1
        if frame_idx % int(fps * 30) == 0:
            print(f"  ... {frame_idx}/{max_frames} frames ({frame_idx/fps/60:.1f} min)")

        if out_video:
            out_video.write(frame)

    cap.release()
    if out_video:
        out_video.release()

    # ---- Resultados ----
    events = events_det.get_events()
    summary = stats.summary()

    # dedup eventos GOAL (múltiples frames)
    events = dedup_events(events)

    report = {
        "video": video_path,
        "resolution": [width, height],
        "fps": fps,
        "total_frames": frame_idx,
        "duration_sec": round(frame_idx / fps, 1),
        "events": events,
        "stats": summary,
        "highlights": [],
    }

    # highlights
    if not args.no_highlights:
        print("Generando highlights...")
        clips = generate_highlights(video_path, [e for e in events if e["type"] in
                                                 ("GOAL", "SHOT")],
                                    out_dir=HIGHLIGHTS_DIR)
        report["highlights"] = clips

    # guardar informe
    stem = Path(video_path).stem
    report_path = OUTPUTS_DIR / f"{stem}_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"\nInforme guardado en: {report_path}")
    print("\n=== RESUMEN ===")
    print(f"Eventos detectados: {len(events)}")
    for ev in events:
        print(f"  [{ev['time']:.1f}s] {ev['type']}")
    print("\nEquipos:")
    for team, st in summary["teams"].items():
        print(f"  {team}: {st['players']} jugadores, {st['total_distance_m']}m recorridos, "
              f"{st['possession_sec']:.0f}s posesión")
    return report


def dedup_events(events, window_s: float = 5.0):
    """Agrupa eventos por ventana temporal, manteniendo el más importante.

    El BallTracker propaga la pelota entre muestras generando ráfagas de
    eventos de la misma jugada (SHOT+BALL_OUT+CORNER en 2s). En lugar de
    deduplicar por tipo (que falla porque se alternan), agrupa TODO lo que
    cae en la misma ventana de tiempo y conserva el evento de mayor prioridad
    (GOAL > SHOT > CORNER > PASS > BALL_OUT). Así 1 jugada = 1 highlight.
    """
    PRIORITY = {"GOAL": 0, "SHOT": 1, "CORNER": 2, "PASS": 3, "BALL_OUT": 4}
    ordered = sorted(events, key=lambda e: e["time"])
    groups = []  # listas de eventos por ventana
    for ev in ordered:
        if groups and ev["time"] - groups[-1][0]["time"] < window_s:
            groups[-1].append(ev)
        else:
            groups.append([ev])
    out = []
    for g in groups:
        best = min(g, key=lambda e: PRIORITY.get(e["type"], 9))
        out.append(best)
    return out


def _annotate(frame, tracked, team_map, ball_m):
    """Dibuja bboxes, IDs y equipos sobre el frame (solo para el video anotado).
    Colores: A=verde, B=rojo, REF=cian, OUT=gris (entrenadores/suplentes),
    UNK=magenta (sin equipo confiable)."""
    for o in tracked:
        x1, y1, x2, y2 = map(int, o.bbox)
        team = team_map.get(o.track_id, "?")
        color = {"A": (0, 255, 0), "B": (0, 0, 255),
                 "REF": (255, 255, 0), "OUT": (128, 128, 128),
                 "UNK": (255, 0, 255)}.get(team, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{o.track_id} {team}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


if __name__ == "__main__":
    main()
