# Falcon Analytics — Football Drone Viz

Analítica de fútbol amateur 100% local para videos de dron (vista cenital).
Ejecutable offline en RTX 4070 Laptop (CUDA) con Python 3.11 + YOLOv8/v5.

## Pipeline

```
python run_pipeline.py videos/partido.mp4 --imgsz 576 --teams siglip
```

Etapas: detección → tracking (BoT-SORT) → clasificación equipos (SigLIP/HVS) → homografía dinámica (ECC) → eventos → estadísticas → highlights.

Outputs:
- `outputs/<video>_annotated_vYYYYMMDD-vNNN.mp4` — video anotado (numeración correlativa)
- `outputs/<video>_report.json` — eventos + stats + highlights
- `highlights/*.mp4` — clips de cada tiro/gol

## Setup

```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt
python run_calibrate.py videos/partido.mp4  # calibrar cancha/arcos → calib/calib.json
```

## Modelos

Football-Players-Tracking (Darkmyter/Roboflow) en `models/football_weights/`:
- yolov8m-640 (jugadores, rápido)
- yolov5m-1280 (pelota, doble pase)
