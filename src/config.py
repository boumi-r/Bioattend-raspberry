# ============================================================
# src/config.py
# Configuration centralisée du projet BioAttend Pi
# ============================================================
import os
from dotenv import load_dotenv

load_dotenv()

# ── Serveur Django ───────────────────────────────────────────
SERVER_URL         = os.getenv("SERVER_URL", "http://192.168.1.100:8000")
API_ENDPOINT       = f"{SERVER_URL.rstrip('/')}/api/face/analyze/"
API_TOKEN          = os.getenv("API_TOKEN", "")  # Used as: Authorization: Bearer {API_TOKEN}

# ── Seuils reconnaissance ────────────────────────────────────
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.60"))

# ── Liveness ─────────────────────────────────────────────────
EAR_THRESHOLD        = float(os.getenv("EAR_THRESHOLD", "0.25"))
BLINK_COUNT_REQUIRED = int(os.getenv("BLINK_COUNT_REQUIRED", "1"))
LIVENESS_TIMEOUT     = int(os.getenv("LIVENESS_TIMEOUT", "5"))

# ── PIR ──────────────────────────────────────────────────────
GPIO_PIR = int(os.getenv("GPIO_PIR", "17"))

# ── Caméra ───────────────────────────────────────────────────
CAMERA_WIDTH  = int(os.getenv("CAMERA_WIDTH",  "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_WARMUP = 2

# ── Timing ───────────────────────────────────────────────────
DEBOUNCE_DELAY = 3   # secondes entre deux détections PIR

# ── Debug ────────────────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# ── Validation ───────────────────────────────────────────────
def validate_config():
    errors = []

    if not SERVER_URL:
        errors.append("SERVER_URL non défini dans .env")

    if DISTANCE_THRESHOLD <= 0 or DISTANCE_THRESHOLD >= 1:
        errors.append(f"DISTANCE_THRESHOLD invalide : {DISTANCE_THRESHOLD}")

    if errors:
        for e in errors:
            print(f"[ERREUR CONFIG] {e}")
        raise ValueError("Configuration invalide — vérifie ton .env")

    if DEBUG:
        print("[CONFIG] Configuration chargée")
        print(f"  SERVER_URL         : {SERVER_URL}")
        print(f"  API_ENDPOINT       : {API_ENDPOINT}")
        print(f"  DISTANCE_THRESHOLD : {DISTANCE_THRESHOLD}")
        print(f"  EAR_THRESHOLD      : {EAR_THRESHOLD}")
        print(f"  GPIO_PIR           : {GPIO_PIR}")