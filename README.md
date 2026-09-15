# Falcon Analytics — Football Drone Viz

Analítica de fútbol amateur 100% local para videos de dron (vista cenital).
Ejecutable offline en RTX 4070 Laptop (CUDA) con Python 3.11 + YOLOv8/v5.

## Pipeline

```
python run_pipeline.py videos/partido.mp4 --imgsz 576 --teams siglip
```

Etapas: detección → tracking (BoT-SORT) → clasificación equipos (SigLIP/HVS) → homografía dinámica (ECC) → eventos → estadísticas → highlights.

Outputs:
- `outputs/<video>_annotated_vYYYYMMDD-vNNN.mp4` — video anotado (numeración correlativa, círculos estilo FIFA)
- `outputs/<video>_report.json` — eventos + stats + highlights
- `outputs/<video>_trayectorias.csv` — posiciones métricas (frame, id, equipo, x_m, y_m)
- `highlights/*.mp4` — clips de cada tiro/gol

## Gemelo digital táctico (Blender)

El CSV de trayectorias se convierte en escena 3D animada (terreno FIFA + cilindros
por jugador + keyframes) ejecutando `tools/blender_tactical_twin.py` dentro de
Blender (editor de texto). Ver el docstring del script para instrucciones.

## Autocalibración por keypoints (opcional)

El pipeline usa calibración manual (`calib/calib.json`). Para autocalibración
dinámica por frame con el modelo Roboflow football-field-detection-f07vi
(`src/autocalib_keypoints.py`), definir la variable de entorno `ROBOFLOW_API_KEY`
antes de correr. Sin key, se usa la calibración manual existente.

## Setup

```bash
uv venv .venv --python 3.11
uv pip install -r requirements.txt
python run_calibrate.py videos/partido.mp4  # calibrar cancha/arcos → calib/calib.json
```

## Modelos

Football-Players-Tracking (Darkmyter/Roboflow) en `models/football_weights/`:
- `yolov8m-640-football-players.pt` — jugadores (rápido, YOLOv8 compatible)
- `yolov8l-640-football-players.pt` — jugadores (preciso)
- `yolov5m-1280-football-players.pt` — pelota (YOLOv5 legacy, NO compatible con YOLOv8)
- `yolov5x-1280-football-players.pt` — pelota (YOLOv5 legacy)

**IMPORTANTE sobre la pelota:** el modelo `yolov5m-1280` (formato YOLOv5) no carga
en Ultralytics YOLOv8 (`TypeError: NOT forwards compatible`). El `BallTracker`
usa `yolov8m-640` corrido a `imgsz=1280` como workaround: la pelota se detecta
bien a 1280px (conf ~0.17–0.44 en tests) pero NO a 640px (0%).
