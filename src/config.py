# ============================================================
# src/config.py
# Configuration centralisée du projet BioAttend Pi
# Toutes les constantes et paramètres sont ici
# ============================================================
import os
from dotenv import load_dotenv

# Charger le fichier .env automatiquement
load_dotenv()


# ── Serveur Django ───────────────────────────────────────────
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

# ── Pins GPIO (numérotation BCM) ─────────────────────────────
GPIO_PIR        = int(os.getenv("GPIO_PIR", "17"))
GPIO_LED_GREEN  = int(os.getenv("GPIO_LED_GREEN", "27"))
GPIO_LED_RED    = int(os.getenv("GPIO_LED_RED", "22"))
GPIO_BUZZER     = int(os.getenv("GPIO_BUZZER", "23"))

# ── Caméra ───────────────────────────────────────────────────
CAMERA_WIDTH    = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT   = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_WARMUP   = 2   # secondes de chauffe caméra avant capture
PICAMERA_INDEX  = int(os.getenv("PICAMERA_INDEX", "0"))

# ── Timing ───────────────────────────────────────────────────
DEBOUNCE_DELAY  = 3   # secondes entre deux détections PIR
LED_DURATION    = 3   # secondes d'allumage LED après décision
BUZZER_SHORT    = 0.2 # secondes buzzer court (refus)
BUZZER_LONG     = 1.0 # secondes buzzer long (accès OK)

# ── Debug ────────────────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# ── Validation au démarrage ──────────────────────────────────
# Vérifie que les paramètres critiques sont bien définis
def validate_config():
    """
    Appelée au démarrage pour vérifier la configuration.
    Lève une exception si un paramètre critique est manquant.
    """
    errors = []

    if not SERVER_URL:
        errors.append("SERVER_URL non défini dans .env")

    if not API_TOKEN:
        errors.append("API_TOKEN non défini dans .env")

    if DISTANCE_THRESHOLD <= 0 or DISTANCE_THRESHOLD >= 1:
        errors.append(f"DISTANCE_THRESHOLD invalide : {DISTANCE_THRESHOLD} (doit être entre 0 et 1)")

    if PICAMERA_INDEX < 0:
        errors.append(f"PICAMERA_INDEX invalide : {PICAMERA_INDEX} (doit être >= 0)")

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
        print(f"  GPIO_LED_GREEN    : {GPIO_LED_GREEN}")
        print(f"  GPIO_LED_RED      : {GPIO_LED_RED}")
        print(f"  GPIO_BUZZER       : {GPIO_BUZZER}")
        print(f"  PICAMERA_INDEX    : {PICAMERA_INDEX}")