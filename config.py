"""Configuración central del pipeline de fútbol por dron."""
from pathlib import Path

# Raíz del proyecto
ROOT = Path(__file__).resolve().parent

# Paths
VIDEOS_DIR = ROOT / "videos"
OUTPUTS_DIR = ROOT / "outputs"
HIGHLIGHTS_DIR = ROOT / "highlights"
CALIB_DIR = ROOT / "calib"
for d in (VIDEOS_DIR, OUTPUTS_DIR, HIGHLIGHTS_DIR, CALIB_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Modelos YOLO (ultralytics) ----
# Modelo de FÚTBOL (entrenado con dataset Roboflow: ball/goalkeeper/player/referee)
# Descargado de https://github.com/Darkmyter/Football-Players-Tracking (Google Drive)
# Mejor que COCO genérico: detecta referee como clase propia y la pelota mejor.
# Alternativas disponibles en models/football_weights/:
#   yolov8m-640-football-players.pt (rápido), yolov8l-640 (preciso),
#   yolov5x-1280 (mejor pelota pero lento), yolov5m-1280
YOLO_MODEL = "models/football_weights/yolov8m-640-football-players.pt"
YOLO_MODEL_COCO = "yolov8m.pt"  # fallback COCO genérico

# Clases del modelo de fútbol (Roboflow football-players-detection)
CLASS_BALL = 0
CLASS_GOALKEEPER = 1
CLASS_PLAYER = 2
CLASS_REFEREE = 3

# Clases COCO (si se usa el modelo COCO)
CLASS_PERSON = 0
CLASS_SPORTS_BALL = 32

# Umbrales de detección
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.50

# ---- Doble pase: pelota (objeto pequeño, necesita resolución alta) ----
# La pelota desde la altura del dron NO se detecta a 640px (0%).
# A 1280px con yolov8m sí (~0.44 conf en tests). Corremos el modelo de pelota
# (yolov8m-640, compatible con YOLOv8) a imgsz=1280 sobre todo el frame cada N
# frames y entre muestras propagamos la última posición con velocidad.
# NOTA: el modelo yolov5m-1280 NO es compatible con Ultralytics YOLOv8
# (error: "NOT forwards compatible"). Se reemplazó por yolov8m-640 a 1280px.
BALL_MODEL = "models/football_weights/yolov8m-640-football-players.pt"
BALL_IMGSZ = 1280
BALL_CONF = 0.15
BALL_SAMPLE_EVERY = 15   # cada N frames se corre el modelo de pelota a 1280px
BALL_STALE_FRAMES = 30   # frames sin detección tras los cuales se suelta la pelota
# Propagación "corta" tras detección real que SÍ alimenta eventos (jugada en curso)
BALL_SHORT_PROP_FRAMES = 45

# ---- Tracking ----
TRACK_BUFFER = 60          # frames que sobrevive una ID sin detecciones
TRACK_MIN_CONF = 0.25      # confianza mínima para entrar al tracker
# Un track con menos frames que esto es ruido (detección espuria) y NO cuenta
# como jugador en las estadísticas. 60 frames @ 60fps = 1 segundo.
MIN_TRACK_FRAMES = 60

# ---- Eventos ----
# Segundos antes/después de un evento para recortar el highlight
HIGHLIGHT_PRE_SEC = 5.0
HIGHLIGHT_POST_SEC = 10.0
# Distancia mínima (m) que debe recorrer la pelota para contar como pase
PASS_DISTANCE_M = 8.0
# Velocidad de pelota (m/s) para considerarla un "tiro"
SHOT_SPEED_MPS = 12.0
# Radio (m) del área del arco para detectar gol (cruzó la línea)
GOAL_TOLERANCE_M = 1.5

# ---- Pitch (medidas reales en metros, FIFA) ----
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
# Se calibra una sola vez por cámara/video (ver run_calibrate.py)
CALIB_FILE = CALIB_DIR / "calib.json"

# ---- Salidas ----
FPS_SAMPLING = 1          # 1 = procesar todos los frames (0 = solo detecciones)
