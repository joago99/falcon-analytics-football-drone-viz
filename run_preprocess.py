"""Preprocesa un video 4K HEVC de dron a una resolución de trabajo (1080p H.264)
que OpenCV/YOLO puedan leer rápido. El archivo de trabajo es la fuente real de
calibración y procesamiento, garantizando consistencia de homografía.

Uso:
  python run_preprocess.py "C:/Analisis/Futbol/DJI_....MP4" [--width 1920]
Genera: work/<nombre>_work.mp4
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from config import OUTPUTS_DIR

WORK_DIR = OUTPUTS_DIR / "work"
WORK_DIR.mkdir(parents=True, exist_ok=True)


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg no está en el PATH. Agrégalo o ajusta FFMPEG_BIN.")
    return exe


def preprocess(video_path: str, width: int = 1920) -> str:
    src = Path(video_path)
    out = WORK_DIR / f"{src.stem}_work.mp4"
    if out.exists():
        print(f"Ya existe: {out}")
        return str(out)
    print(f"Transcodificando {src.name} -> {out} ({width}px, H.264)...")
    cmd = [
        _ffmpeg(), "-y", "-i", str(src),
        "-map", "0:0",                    # solo el stream HEVC principal
        "-vf", f"scale={width}:-2",       # mantener aspecto, par
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",            # compatible con OpenCV
        "-an",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"Error de transcodificación:\n{r.stderr[-1500:]}")
    print(f"Listo: {out} ({out.stat().st_size/1e6:.0f} MB)")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Preprocesar video 4K a resolución de trabajo")
    ap.add_argument("video", help="Ruta al video")
    ap.add_argument("--width", type=int, default=1920, help="Ancho de trabajo (px)")
    args = ap.parse_args()
    preprocess(args.video, args.width)
