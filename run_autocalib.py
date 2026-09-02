"""Calibración AUTOMÁTICA de la cancha para vista cenital de dron.

Detecta el campo (máscara verde) → toma el contorno mayor → calcula las
4 esquinas en orden → homografía píxel→metro (105x68) → estima los postes de
cada arco en el centro de las líneas de gol (proyectando metros→píxel con la
homografía inversa).

Útil para un primer pipeline automático; luego se puede refinar manualmente
con run_calibrate.py si la cancha no se detecta perfecta.

Uso:
  python run_autocalib.py "C:/.../work_video.mp4" [--second 5] [--frame PNG]
Genera calib/calib.json (igual que la calibración manual).
"""
import argparse
import json
import sys

import cv2
import numpy as np

from config import CALIB_FILE, PITCH_LENGTH, PITCH_WIDTH

# Medias del arco FIFA (7.32 m de ancho)
GOAL_WIDTH_M = 7.32


def detect_pitch_corners(frame):
    """Detecta las 4 esquinas de la cancha (vista cenital) de forma robusta.

    Estrategia:
      1. Máscara de césped (verde) -> región del terreno de juego.
      2. Dentro de esa región, máscara de LÍNEAS BLANCAS del campo.
      3. El rectángulo exterior (min/max x e y) de las líneas blancas define
         los límites del campo de juego (touchlines + goal lines).
      4. Devolver esas 4 esquinas ordenadas.

    Es robusto porque las gradas/autos/líneas ajenas quedan fuera de la máscara
    verde y no contaminan el rectángulo.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 1) césped (verde)
    green = ((h >= 35) & (h <= 90) & (s > 40) & (v > 50)).astype(np.uint8) * 255
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    # quedarnos con el blob verde más grande (el campo, no zonas externas verdes)
    contours, _ = cv2.findContours(green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, green
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < frame.shape[0] * frame.shape[1] * 0.10:
        return None, green
    green_mask = np.zeros_like(green)
    cv2.drawContours(green_mask, [c], -1, 255, -1)

    # 2) líneas blancas dentro del césped
    white = ((v > 150) & (s < 110)).astype(np.uint8) * 255
    lines_field = cv2.bitwise_and(white, green_mask)

    ys, xs = np.nonzero(lines_field)
    if len(xs) < 50:
        return None, lines_field
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    # validar que el rectángulo sea razonable (no degenerado ni minúsculo)
    if (x_max - x_min) < frame.shape[1] * 0.25 or (y_max - y_min) < frame.shape[0] * 0.25:
        return None, lines_field

    corners = np.array([[x_min, y_min], [x_max, y_min],
                        [x_max, y_max], [x_min, y_max]], dtype=np.float32)
    return order_corners(corners), lines_field


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 esquinas en sentido horario [TL, TR, BR, BL]."""
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def compute_homography(corners_ordered: np.ndarray) -> np.ndarray:
    """Homografía píxel -> metros sobre cancha FIFA 105x68."""
    src = corners_ordered.astype(np.float32)
    dst = np.array([
        [0, 0],
        [PITCH_LENGTH, 0],
        [PITCH_LENGTH, PITCH_WIDTH],
        [0, PITCH_WIDTH],
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H


def estimate_goals(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estima los postes de cada arco (en píxel) en el centro de las líneas de gol.

    Arco izquierdo en metros: (0, center ± GOAL_WIDTH/2).
    Arco derecho  en metros: (PITCH_LENGTH, center ± GOAL_WIDTH/2).
    Se proyectan a píxel con H_inv.
    """
    H_inv = np.linalg.inv(H)
    cy = PITCH_WIDTH / 2.0
    half = GOAL_WIDTH_M / 2.0

    def meters_to_px(mx, my):
        p = np.array([mx, my], dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(p, H_inv).reshape(-1, 2)[0]
        return out

    goalL = np.array([meters_to_px(0, cy - half), meters_to_px(0, cy + half)])
    goalR = np.array([meters_to_px(PITCH_LENGTH, cy - half),
                      meters_to_px(PITCH_LENGTH, cy + half)])
    return goalL.astype(np.float32), goalR.astype(np.float32)


def autocalibrate_frame(frame) -> dict:
    corners = detect_pitch_corners(frame)[0]
    if corners is None:
        raise RuntimeError("No se pudo detectar el campo (máscara verde). "
                           "Usa calibración manual: python run_calibrate.py")
    H = compute_homography(corners)
    goalL, goalR = estimate_goals(H)
    return {
        "video": "",
        "frame_used": None,
        "corners_px": corners.tolist(),
        "corners_ordered_px": corners.tolist(),
        "goal_left_px": goalL.tolist(),
        "goal_right_px": goalR.tolist(),
        "homography": H.tolist(),
        "pitch_length": PITCH_LENGTH,
        "pitch_width": PITCH_WIDTH,
        "automatic": True,
    }


def main():
    ap = argparse.ArgumentParser(description="Auto-calibración de cancha (vista cenital)")
    ap.add_argument("input", help="Video de trabajo o PNG")
    ap.add_argument("--second", type=float, default=5.0, help="Segundo del video")
    args = ap.parse_args()

    if args.input.lower().endswith((".png", ".jpg", ".jpeg")):
        frame = cv2.imread(args.input)
        if frame is None:
            sys.exit(f"No se pudo leer imagen: {args.input}")
    else:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            sys.exit(f"No se pudo abrir: {args.input}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        cap.set(cv2.CAP_PROP_POS_MSEC, args.second * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit("No se pudo leer un frame del video.")

    calib = autocalibrate_frame(frame)
    CALIB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Auto-calibración guardada en {CALIB_FILE}")
    print(f"  Esquinas (px): {[[int(a), int(b)] for a, b in calib['corners_px']]}")
    print(f"  Arco izq (px): {[[int(a), int(b)] for a, b in calib['goal_left_px']]}")
    print(f"  Arco der (px): {[[int(a), int(b)] for a, b in calib['goal_right_px']]}")
    # guardar overlay de verificación
    out = frame.copy()
    for p in calib["corners_px"]:
        cv2.circle(out, (int(p[0]), int(p[1])), 8, (0, 255, 0), -1)
    for p in list(calib["goal_left_px"]) + list(calib["goal_right_px"]):
        cv2.circle(out, (int(p[0]), int(p[1])), 8, (0, 0, 255), -1)
    overlay_path = CALIB_FILE.parent / "autocalib_check.png"
    cv2.imwrite(str(overlay_path), out)
    print(f"  Overlay de verificación: {overlay_path}")
    return calib


if __name__ == "__main__":
    main()
