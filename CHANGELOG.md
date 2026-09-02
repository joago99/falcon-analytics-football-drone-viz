# CHANGELOG

All notable changes to Falcon Analytics — Football Drone Viz.

## [v0.2.1] — 2026-09-03

### Fixed
- **BallTracker reescrito**: reemplazado crop+upscale (que no detectaba pelota a 640px)
  por full-frame a 1280px con yolov8m. La pelota no se detecta a 640px (0%) pero
  sí a 1280px (conf ~0.17–0.44). El crop+upscale borronea detalle insuficiente.
- **BALL_MODEL**: cambiado de `yolov5m-1280` (incompatible con YOLOv8, lanzaba
  TypeError) a `yolov8m-640` (compatible).
- **BALL_IMGSZ**: 640 → 1280 (resolución requerida para detección de pelota).
- **BALL_CONF**: 0.20 → 0.15 (ajuste para captar detecciones débiles de pelota).

### Added
- Documentación del workaround YOLOv5→YOLOv8 en README.md.

## [v0.2.0] — 2026-09-03

### Changed
- **BallTracker** reescrito: ahora hace crop+upscale de la región alrededor de la pelota (4x) en lugar de modelo separado a 1280px.
  - Soluciona incompatibilidad YOLOv5 (yolov5m-1280) con Ultralytics YOLOv8.
  - `BALL_MODEL` ahora apunta a `yolov8m-640-football-players.pt` (compatible).
  - Mejor detección de pelota en vista cenital de dron.
- **Numeración correlativa**: videos anotados llevan sufijo `annotated_vYYYYMMDD-vNNN.mp4`.
- **Calibración**: eliminado import muerto `Path` en `src/calibrate.py`.

### Fixed
- `BALL_MODEL` apuntaba a modelo YOLOv5 incompatible con YOLOv8.

## [v0.1.0] — 2026-09-02

### Added
- Pipeline completo: detección → tracking (BoT-SORT) → equipos (SigLIP/HVS) → homografía dinámica (ECC) → eventos → estadísticas → highlights.
- Doble pase para detección de pelota (alta resolución via crop).
- Compensación de movimiento de cámara (ECC) para dron con pan lateral.
- Clasificación de equipos con SigLIP (embeddings) + KMeans.
- Filtro temporal de árbitros (top-3 oficiales consistentes).
- Filtro espacial: staff/entrenadores fuera de cancha excluidos de stats.
