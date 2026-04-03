
import sys
import os
import time
import logging

sys.path.insert(0, os.path.dirname(__file__))

import config
import pir
import camera
import liveness
import face_recognition
import api_client
import gpio_feedback


logging.basicConfig(
    level   = logging.DEBUG if config.DEBUG else logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(name)s : %(message)s",
    handlers=[
        logging.StreamHandler(),                   
        logging.FileHandler("bioattend.log"),       
    ]
)
logger = logging.getLogger(__name__)


def main():
    
    logger.info("=" * 55)
    logger.info("BIOATTEND — Démarrage du système")
    logger.info("=" * 55)

    try:
        config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration invalide : {e}")
        sys.exit(1)

    gpio_feedback.setup()
    pir.setup()

    try:
        camera.open_camera()
        camera.pause_camera()
    except Exception as e:
        logger.error(f"Impossible d'ouvrir la caméra : {e}")
        gpio_feedback.signal_error()
        sys.exit(1)

    # Charger le modèle InsightFace
    try:
        face_recognition.init_model()
    except Exception as e:
        logger.error(f"Impossible de charger InsightFace : {e}")
        gpio_feedback.signal_error()
        sys.exit(1)

    logger.info("Vérification connexion serveur Django...")
    if not api_client.check_server():
        logger.error("Serveur Django inaccessible, vérifier le réseau")
        gpio_feedback.signal_error()
    else:
        logger.info("Serveur Django accessible")

    gpio_feedback.signal_ready()
    logger.info("Système prêt — en attente de présence...")
    logger.info("-" * 55)

    try:
        while True:
            _process_one_detection()
            time.sleep(config.DEBOUNCE_DELAY)

    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur (Ctrl+C)")

    finally:
        camera.close_camera()
        gpio_feedback.cleanup()
        logger.info("Système arrêté proprement")


def _process_one_detection():

    logger.info("En attente de présence (PIR)...")
    detected = pir.wait_for_motion()

    if not detected:
        return

    logger.info("Présence détectée !")
    gpio_feedback.signal_processing()

    camera.mouvement_camera()

    # Étape 1 : Liveness detection
    logger.info("Liveness detection en cours...")

    video_stream = camera.get_video_stream()

    if video_stream is not None:
        liveness_result = liveness.check_liveness_realtime(video_stream)
    else:
        logger.warning("Pas de flux vidéo — analyse image statique")
        image_bytes     = camera.capture_image_opencv()
        liveness_result = liveness.check_liveness_opencv(image_bytes)

    logger.info(f"Liveness résultat : {liveness_result}")

    if not liveness_result["is_live"]:
        logger.warning(f"SPOOF détecté : {liveness_result['reason']}")
        gpio_feedback.signal_spoof_detected()
        camera.pause_camera()
        return

    logger.info("Liveness validée — personne vivante confirmée")

    # Étape 2 : Capture image pour reconnaissance
    logger.info("Capture image pour reconnaissance...")
    try:
        image_bytes = camera.capture_image_opencv()
    except Exception as e:
        logger.error(f"Erreur capture : {e}")
        gpio_feedback.signal_error()
        camera.pause_camera()
        return

    # Étape 3 : Extraction embedding InsightFace
    logger.info("Extraction embedding InsightFace...")
    face_result = face_recognition.get_embedding(image_bytes)

    if not face_result.get("success"):
        logger.warning(f"Pas de visage détecté : {face_result.get('error')}")
        gpio_feedback.signal_access_denied()
        camera.pause_camera()
        return

    embedding = face_result["embedding"]
    bbox      = face_result["bbox"]
    logger.info(f"Embedding extrait : {len(embedding)} valeurs, bbox={bbox}")

    # Étape 4 : Envoi embedding à l'API Django /api/face/identify/
    logger.info("Envoi embedding au serveur Django...")
    result = api_client.send_embedding(embedding, bbox)

    if not result.get("success"):
        error = result.get("error", "Erreur inconnue")
        logger.warning(f"API erreur : {error}")

        if "inaccessible" in error or "Timeout" in error:
            gpio_feedback.signal_error()
        else:
            gpio_feedback.signal_access_denied()
        camera.pause_camera()
        return

    # Étape 5 : Traitement réponse identification
    name    = result.get("name", "Inconnu")
    status  = result.get("status", "")
    logger.info(f"Identification : {name} — {status}")

    gpio_feedback.signal_access_granted()
    logger.info("-" * 55)

    camera.pause_camera()

if __name__ == "__main__":
    main()