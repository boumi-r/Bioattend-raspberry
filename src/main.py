# ============================================================
# src/main.py
# Rôle : script principal du système BioAttend Pi
#
# Flux complet :
#   1. PIR détecte présence
#   2. Caméra capture image
#   3. Anti-spoof prediction (MiniFASNet)
#   4. Envoi image au serveur Django
#   5. Décision accès + message écran
#   6. Journalisation
#
# Lancer avec : python src/main.py
# ============================================================
import sys
import os
import time
import logging
import cv2
import numpy as np

# Ajouter src au path
sys.path.insert(0, os.path.dirname(__file__))

import config
import pir
import camera
import api_client
import gpio_feedback
from anti_spoof_predict import AntiSpoofPredict

# ── Configuration du logging ─────────────────────────────────
logging.basicConfig(
    level   = logging.DEBUG if config.DEBUG else logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s : %(message)s",
    handlers=[
        logging.StreamHandler(),                    # terminal
        logging.FileHandler("bioattend.log"),       # fichier log
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Boucle principale du système BioAttend.
    Tourne indéfiniment jusqu'à Ctrl+C.
    """
    logger.info("=" * 55)
    logger.info("BIOATTEND — Démarrage du système")
    logger.info("=" * 55)

    # ── 1. Validation configuration ──────────────────────────
    try:
        config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration invalide : {e}")
        sys.exit(1)

    # ── 2. Initialisation feedback écran ─────────────────────
    gpio_feedback.setup()
    pir.setup()

    # ── 3. Initialisation anti-spoof predict ─────────────────
    try:
        anti_spoof = AntiSpoofPredict(device_id=0)
        logger.info("AntiSpoofPredict initialisé")
    except Exception as e:
        logger.error(f"Impossible d'initialiser AntiSpoofPredict : {e}")
        gpio_feedback.signal_error()
        sys.exit(1)

    # ── 4. Initialisation caméra ─────────────────────────────
    cam = camera.CameraManager()
    try:
        cam.open()
    except Exception as e:
        logger.error(f"Impossible d'ouvrir la caméra : {e}")
        gpio_feedback.signal_error()
        sys.exit(1)

    # ── 5. Vérification serveur Django ───────────────────────
    logger.info("Vérification connexion serveur Django...")
    if not api_client.check_server():
        logger.error("Serveur Django inaccessible — vérifier le réseau")
        gpio_feedback.signal_error()
        # On continue quand même — le serveur peut redémarrer
    else:
        logger.info("Serveur Django accessible")

    # ── 6. Signal prêt ───────────────────────────────────────
    gpio_feedback.signal_ready()
    logger.info("Système prêt — en attente de présence...")
    logger.info("-" * 55)

    # ── 7. Boucle principale ─────────────────────────────────
    try:
        while True:
            _process_one_detection(cam, anti_spoof)
            # Délai anti-rebond — évite double détection
            time.sleep(config.DEBOUNCE_DELAY)

    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur (Ctrl+C)")

    finally:
        # Nettoyage propre à la sortie
        cam.close()
        gpio_feedback.cleanup()
        logger.info("Système arrêté proprement")


def _process_one_detection(cam: camera.CameraManager, anti_spoof: AntiSpoofPredict):
    """
    Traite une détection complète :
    PIR → Caméra → Anti-Spoof → API → Décision

    Paramètres :
        cam : instance CameraManager initialisée
        anti_spoof : instance AntiSpoofPredict initialisée
    """

    # ── ÉTAPE 1 : Attendre détection PIR ────────────────────
    logger.info("En attente de présence (PIR)...")
    detected = pir.wait_for_motion()

    if not detected:
        return

    logger.info("Présence détectée !")
    gpio_feedback.signal_processing()

    # ── ÉTAPE 2 : Anti-Spoof Prediction ────────────────────────────────────────────
    logger.info("Anti-spoof prediction en cours...")
    
    try:
        image_bytes = cam.capture_image()
    except Exception as e:
        logger.error(f"Erreur capture image pour anti-spoof : {e}")
        gpio_feedback.signal_error()
        return
    
    # Convertir bytes en image OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        logger.error("Image invalide pour anti-spoof")
        gpio_feedback.signal_spoof_detected()
        return
    
    # Charger le meilleur modèle disponible
    model_path = None
    for model_file in os.listdir("./models"):
        if model_file.endswith(".pth"):
            model_path = os.path.join("./models", model_file)
            logger.info(f"Utilisation du modèle : {model_file}")
            break
    
    if model_path is None:
        logger.error("Aucun modèle .pth trouvé dans ./models")
        gpio_feedback.signal_error()
        return
    
    # Prédiction anti-spoof
    try:
        result = anti_spoof.predict(img, model_path)
        # result est un array softmax [fake_score, real_score]
        fake_score = float(result[0][0])
        real_score = float(result[0][1])
        logger.info(f"Anti-spoof scores - Fake: {fake_score:.4f}, Real: {real_score:.4f}")
        
        # Seuil : si score réel > 0.5, visage vivant
        if real_score < 0.5:
            logger.warning(f"SPOOF détecté (score réel: {real_score:.4f})")
            gpio_feedback.signal_spoof_detected()
            return
        
        logger.info(f"Liveness validée - Visage vivant confirmé (score: {real_score:.4f})")
    except Exception as e:
        logger.error(f"Erreur lors de anti-spoof prediction : {e}")
        gpio_feedback.signal_error()
        return

    # ── ÉTAPE 3 : Image déjà capturée lors de l'anti-spoof ────────────
    logger.info("Image capturée — passage à l'envoi API")

    # ── ÉTAPE 4 : Envoi à l'API Django ──────────────────────
    logger.info("Envoi image au serveur Django...")
    result = api_client.send_image(image_bytes)

    # ── ÉTAPE 5 : Traitement réponse ────────────────────────
    if not result.get("success"):
        error = result.get("error", "Erreur inconnue")
        logger.warning(f"API erreur : {error}")

        if "inaccessible" in error or "Timeout" in error:
            gpio_feedback.signal_error()
        else:
            # Aucun visage détecté par InsightFace
            gpio_feedback.signal_access_denied()
        return

    # ── ÉTAPE 6 : Décision accès ─────────────────────────────
    embedding = result.get("embedding", [])
    bbox      = result.get("bbox", [])

    logger.info(f"Embedding reçu : {len(embedding)} valeurs")
    logger.info(f"Bbox visage    : {bbox}")

    # Pour l'instant on log l'embedding et on autorise
    # La comparaison avec la BDD sera ajoutée en Phase 3
    logger.info("Embedding extrait avec succès")
    logger.info("Phase 3 : comparaison BDD à implémenter")

    # Message accès autorisé (temporaire — sans comparaison BDD)
    gpio_feedback.signal_access_granted()
    logger.info("-" * 55)


# ── Point d'entrée ───────────────────────────────────────────
if __name__ == "__main__":
    main()