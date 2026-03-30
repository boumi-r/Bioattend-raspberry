# ============================================================
# src/main.py
# Rôle : script principal du système BioAttend Pi
#
# Flux complet :
#   1. PIR détecte présence
#   2. Caméra capture image
#   3. Liveness detection (clignement yeux)
#   4. Envoi image au serveur Django
#   5. Décision accès + LED + Buzzer
#   6. Journalisation
#
# Lancer avec : python src/main.py
# ============================================================
import sys
import os
import time
import logging

# Ajouter src au path
sys.path.insert(0, os.path.dirname(__file__))

import config
import pir
import camera
import liveness
import api_client
import gpio_feedback

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

    # ── 2. Initialisation GPIO ───────────────────────────────
    gpio_feedback.setup()
    pir.setup()

    # ── 3. Initialisation caméra ─────────────────────────────
    cam = camera.CameraManager()
    try:
        cam.open()
    except Exception as e:
        logger.error(f"Impossible d'ouvrir la caméra : {e}")
        gpio_feedback.signal_error()
        sys.exit(1)

    # ── 4. Vérification serveur Django ───────────────────────
    logger.info("Vérification connexion serveur Django...")
    if not api_client.check_server():
        logger.error("Serveur Django inaccessible — vérifier le réseau")
        gpio_feedback.signal_error()
        # On continue quand même — le serveur peut redémarrer
    else:
        logger.info("Serveur Django accessible")

    # ── 5. Signal prêt ───────────────────────────────────────
    gpio_feedback.signal_ready()
    logger.info("Système prêt — en attente de présence...")
    logger.info("-" * 55)

    # ── 6. Boucle principale ─────────────────────────────────
    try:
        while True:
            _process_one_detection(cam)
            # Délai anti-rebond — évite double détection
            time.sleep(config.DEBOUNCE_DELAY)

    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur (Ctrl+C)")

    finally:
        # Nettoyage propre à la sortie
        cam.close()
        gpio_feedback.cleanup()
        logger.info("Système arrêté proprement")


def _process_one_detection(cam: camera.CameraManager):
    """
    Traite une détection complète :
    PIR → Caméra → Liveness → API → Décision

    Paramètre :
        cam : instance CameraManager initialisée
    """

    # ── ÉTAPE 1 : Attendre détection PIR ────────────────────
    logger.info("En attente de présence (PIR)...")
    detected = pir.wait_for_motion()

    if not detected:
        return

    logger.info("Présence détectée !")
    gpio_feedback.signal_processing()

    # ── ÉTAPE 2 : Liveness Detection ────────────────────────
    logger.info("Liveness detection en cours...")

    video_stream = cam.get_video_stream()

    if video_stream is not None:
        # Flux vidéo disponible → liveness temps réel (clignement)
        liveness_result = liveness.check_liveness_realtime(video_stream)
    else:
        # Pas de flux vidéo → capture image + analyse statique
        logger.warning("Pas de flux vidéo — analyse image statique")
        image_bytes     = cam.capture_image()
        liveness_result = liveness.check_liveness_opencv(image_bytes)

    logger.info(f"Liveness résultat : {liveness_result}")

    # Visage non vivant → spoof détecté
    if not liveness_result["is_live"]:
        logger.warning(f"SPOOF détecté : {liveness_result['reason']}")
        gpio_feedback.signal_spoof_detected()
        return

    logger.info("Liveness validée — personne vivante confirmée")

    # ── ÉTAPE 3 : Capture image finale ──────────────────────
    logger.info("Capture image pour reconnaissance...")
    try:
        image_bytes = cam.capture_image()
    except Exception as e:
        logger.error(f"Erreur capture : {e}")
        gpio_feedback.signal_error()
        return

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

    # Signal accès autorisé (temporaire — sans comparaison BDD)
    gpio_feedback.signal_access_granted()
    logger.info("-" * 55)


# ── Point d'entrée ───────────────────────────────────────────
if __name__ == "__main__":
    main()