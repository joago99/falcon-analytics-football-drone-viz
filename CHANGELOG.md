# CHANGELOG

All notable changes to Falcon Analytics — Football Drone Viz.

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
