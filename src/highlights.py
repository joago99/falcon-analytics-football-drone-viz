"""Generación de highlights: cortes de video alrededor de cada evento."""
import subprocess
from pathlib import Path

from config import HIGHLIGHTS_DIR, HIGHLIGHT_PRE_SEC, HIGHLIGHT_POST_SEC


def _ensure_ffmpeg():
    """ffmpeg debe estar en PATH o como binario. Comprueba y devuelve el comando."""
    import shutil
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg no está instalado. Instálalo o agrega la ruta al PATH.")
    return exe


def cut_clip(video_path: str, out_path: str, start_sec: float, end_sec: float):
    """Recorta el video [start_sec, end_sec] a out_path (H.264, sin re-escalado)."""
    ffmpeg = _ensure_ffmpeg()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # re-encode para corte preciso por segundo
    cmd = [
        ffmpeg, "-y", "-i", video_path,
        "-ss", f"{start_sec:.2f}", "-to", f"{end_sec:.2f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def generate_highlights(video_path: str, events: list[dict], out_dir=None):
    """Genera un clip por evento. Devuelve lista de dicts con rutas generadas."""
    out_dir = Path(out_dir or HIGHLIGHTS_DIR)
    clips = []
    for i, ev in enumerate(events):
        t = ev["time"]
        start = max(0.0, t - HIGHLIGHT_PRE_SEC)
        end = t + HIGHLIGHT_POST_SEC
        safe = str(ev["type"]).lower()
        fname = out_dir / f"{i:03d}_{safe}_t{int(t)}s.mp4"
        try:
            cut_clip(video_path, str(fname), start, end)
            clips.append({
                "index": i, "type": ev["type"], "time": round(t, 1),
                "path": str(fname), "start": round(start, 1), "end": round(end, 1),
            })
        except Exception as e:
            clips.append({"index": i, "type": ev["type"], "time": round(t, 1),
                          "error": str(e)})
    return clips
