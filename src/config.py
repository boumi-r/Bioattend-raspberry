
import os
from dotenv import load_dotenv
load_dotenv()


SERVER_URL    = os.getenv("SERVER_URL", "https://bioattend.138.199.195.144.sslip.io")
API_ENDPOINT  = f"{SERVER_URL.rstrip('/')}/api/face/analyze/"
API_TOKEN     = os.getenv("API_TOKEN", "")

# ── Seuils de reconnaissance ─────────────────────────────────
# Distance cosine InsightFace
# < DISTANCE_THRESHOLD = même personne → accès autorisé
# > DISTANCE_THRESHOLD = personne différente → accès refusé
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.60"))

# ── Liveness detection ───────────────────────────────────────
# Seuil EAR (Eye Aspect Ratio)
# Un oeil ouvert = ~0.30 | Un oeil fermé = ~0.15
# Si EAR < EAR_THRESHOLD → clignement détecté
EAR_THRESHOLD       = float(os.getenv("EAR_THRESHOLD", "0.25"))
BLINK_COUNT_REQUIRED = int(os.getenv("BLINK_COUNT_REQUIRED", "1"))
LIVENESS_TIMEOUT    = int(os.getenv("LIVENESS_TIMEOUT", "5"))


GPIO_PIR        = int(os.getenv("GPIO_PIR", "17"))


CAMERA_WIDTH    = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT   = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_WARMUP   = 2   # secondes de chauffe caméra avant capture

DEBOUNCE_DELAY  = 3   # secondes entre deux détections PIR
FEEDBACK_DURATION = float(os.getenv("FEEDBACK_DURATION", "3"))


DEBUG = os.getenv("DEBUG", "True").lower() == "true"


def validate_config():
    
    errors = []

    if not SERVER_URL:
        errors.append("SERVER_URL non défini dans .env")

    if not API_TOKEN:
        errors.append("API_TOKEN non défini dans .env")

    if DISTANCE_THRESHOLD <= 0 or DISTANCE_THRESHOLD >= 1:
        errors.append(f"DISTANCE_THRESHOLD invalide : {DISTANCE_THRESHOLD} (doit être entre 0 et 1)")

    if errors:
        for e in errors:
            print(f"[CONFIG ERREUR] {e}")
        raise ValueError("Configuration invalide — vérifie ton fichier .env")

    if DEBUG:
        print("[CONFIG] Configuration chargée avec succès")
        print(f"  SERVER_URL        : {SERVER_URL}")
        print(f"  API_ENDPOINT      : {API_ENDPOINT}")
        print(f"  DISTANCE_THRESHOLD: {DISTANCE_THRESHOLD}")
        print(f"  EAR_THRESHOLD     : {EAR_THRESHOLD}")
        print(f"  GPIO_PIR          : {GPIO_PIR}")
        print(f"  FEEDBACK_DURATION : {FEEDBACK_DURATION}")