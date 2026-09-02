"""Calibración interactiva de cancha y arcos para un video de dron.

Sobre un frame representativo del video, el usuario hace clic (en orden) en:
  1-2-3-4 : las cuatro esquinas de la cancha (en cualquier orden)
  A1-A2   : la línea del arco izquierdo (los dos postes)
  B1-B2   : la línea del arco derecho (los dos postes)

Con las 4 esquinas se calcula la homografía píxel -> metros (FIFA 105x68).
Con los arcos se saben las líneas de gol para detectar goles y tiros.

Uso:
  python run_calibrate.py videos/partido.mp4
Guarda el resultado en calib/calib.json.
"""
import argparse
import json
import sys

import cv2
import numpy as np

from config import CALIB_FILE, PITCH_LENGTH, PITCH_WIDTH

POINT_NAMES = [
    "esquina 1", "esquina 2", "esquina 3", "esquina 4",
    "poste izq (arco izquierdo)", "poste der (arco izquierdo)",
    "poste izq (arco derecho)", "poste der (arco derecho)",
]


def pick_frame(video_path: str, frame_second: float = 5.0) -> np.ndarray:
    """Extrae un frame representativo (a los N segundos) del video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_MSEC, frame_second * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("No se pudo leer un frame del video.")
    return frame, fps


def select_points(frame: np.ndarray, names: list[str]) -> list[tuple[int, int]]:
    """Ventana OpenCV para que el usuario haga clic en los puntos en orden."""
    pts: list[tuple[int, int]] = []
    disp = frame.copy()
    cv2.namedWindow("Calibracion", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibracion", lambda e, x, y, fl, p: _on_click(e, x, y, pts, names))
    msg = f"Click en {names[0]} (Punto {len(pts)+1}/{len(names)}). ESC=salir, R=deshacer"
    cv2.imshow("Calibracion", disp)
    while True:
        d = disp.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(d, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(d, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # dibujar arcos entre puntos del mismo par
        if len(pts) >= 5:
            cv2.line(d, pts[3], pts[4], (255, 0, 0), 2)
            cv2.line(d, pts[4], pts[5], (255, 0, 0), 2)
        if len(pts) >= 7:
            cv2.line(d, pts[3], pts[6], (0, 255, 255), 2)
            cv2.line(d, pts[6], pts[7], (0, 255, 255), 2)
        if len(pts) == 4:
            cv2.line(d, pts[0], pts[1], (0, 255, 0), 1)
            cv2.line(d, pts[1], pts[2], (0, 255, 0), 1)
            cv2.line(d, pts[2], pts[3], (0, 255, 0), 1)
            cv2.line(d, pts[3], pts[0], (0, 255, 0), 1)
        cv2.putText(d, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.imshow("Calibracion", d)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            sys.exit("Calibracion cancelada por el usuario.")
        if key == ord("r"):
            if pts:
                pts.pop()
                msg = f"Deshecho. Click en {names[len(pts)]}"
        if len(pts) == len(names):
            break
    cv2.destroyAllWindows()
    return pts


def _on_click(event, x, y, flags, pts, names):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(pts) < len(names):
            pts.append((x, y))


def order_corners(corners: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Ordena 4 esquinas en sentido horario empezando por arriba-izquierda."""
    arr = np.array(corners, dtype=np.float32)
    s = arr.sum(axis=1)
    diff = np.diff(arr, axis=1).ravel()
    tl = arr[np.argmin(s)]
    br = arr[np.argmax(s)]
    tr = arr[np.argmin(diff)]
    bl = arr[np.argmax(diff)]
    return [tuple(int(v) for v in p) for p in (tl, tr, br, bl)]


def compute_homography(corners_ordered: list[tuple[int, int]]):
    """Homografía píxel -> metros sobre un campo FIFA 105x68."""
    src = np.array(corners_ordered, dtype=np.float32)
    dst = np.array([
        [0, 0],
        [PITCH_LENGTH, 0],
        [PITCH_LENGTH, PITCH_WIDTH],
        [0, PITCH_WIDTH],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def calibrate_video(video_path: str, frame_second: float = 5.0):
    frame, fps = pick_frame(video_path, frame_second)
    pts = select_points(frame, POINT_NAMES)
    corners_ordered = order_corners(pts[:4])
    goalL = [pts[4], pts[5]]
    goalR = [pts[6], pts[7]]

    H = compute_homography(corners_ordered)
    calib = {
        "video": str(video_path),
        "frame_used": frame_second,
        "fps": fps,
        "corners_px": [list(map(int, p)) for p in pts[:4]],
        "corners_ordered_px": [list(map(int, p)) for p in corners_ordered],
        "goal_left_px": [list(map(int, p)) for p in goalL],
        "goal_right_px": [list(map(int, p)) for p in goalR],
        "homography": H.tolist(),
        "pitch_length": PITCH_LENGTH,
        "pitch_width": PITCH_WIDTH,
    }
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Calibracion guardada en {CALIB_FILE}")
    print(f"  FPS video: {fps:.1f}")
    print(f"  Cancha (px): {calib['corners_ordered_px']}")
    print(f"  Arco izq (px): {calib['goal_left_px']}")
    print(f"  Arco der (px): {calib['goal_right_px']}")
    return calib


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Calibración de cancha y arcos")
    ap.add_argument("video", help="Ruta al video")
    ap.add_argument("--second", type=float, default=5.0,
                    help="Segundo del video a usar como frame de calibración")
    args = ap.parse_args()
    calibrate_video(args.video, args.second)
