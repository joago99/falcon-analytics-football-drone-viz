"""Runner de calibración interactiva de cancha + arcos.

Uso:
  python run_calibrate.py videos/partido.mp4 [--second 5]
"""
import argparse
import sys
from pathlib import Path

from config import VIDEOS_DIR
from src.calibrate import calibrate_video


def main():
    ap = argparse.ArgumentParser(description="Calibración de cancha y arcos")
    ap.add_argument("video", help="Ruta al video (o nombre en videos/)")
    ap.add_argument("--second", type=float, default=5.0,
                    help="Segundo del video a usar para el frame de calibración")
    args = ap.parse_args()

    p = Path(args.video)
    if not p.exists():
        cand = VIDEOS_DIR / p
        if cand.exists():
            p = cand
        else:
            sys.exit(f"No se encontró: {args.video} (probado también en {VIDEOS_DIR})")

    calibrate_video(str(p), args.second)


if __name__ == "__main__":
    main()
