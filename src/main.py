# ============================================================
# src/main.py
# ============================================================
import sys
import os
import time
import logging
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import config
import pir
import api_client
import gpio_feedback
from anti_spoof_predict import AntiSpoofPredict
from embedding_extractor import EmbeddingExtractor
from camera import CameraManager, COLOR_GREEN, COLOR_RED, COLOR_ORANGE, COLOR_WHITE

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
        anti_spoof = AntiSpoofPredict(device_id=0)
        logger.info("AntiSpoofPredict initialisé")
    except Exception as e:
        logger.error(f"Impossible d'initialiser AntiSpoofPredict : {e}")
        sys.exit(1)

    try:
        extractor = EmbeddingExtractor(model_name="buffalo_sc")
        logger.info("EmbeddingExtractor InsightFace initialisé")
    except Exception as e:
        logger.error(f"Impossible d'initialiser InsightFace : {e}")
        sys.exit(1)

    # ── Caméra (ouvre aussi la fenêtre d'affichage) ──────────
    cam = CameraManager()
    try:
        cam.open()
        cam.set_status("Système prêt — approchez-vous...", COLOR_WHITE)
    except Exception as e:
        logger.error(f"Impossible d'ouvrir la caméra : {e}")
        sys.exit(1)

    logger.info("Vérification connexion serveur Django...")
    if not api_client.check_server():
        logger.error("Serveur Django inaccessible")
        cam.set_status("⚠ Serveur inaccessible", COLOR_RED)
    else:
        logger.info("Serveur Django accessible")

    gpio_feedback.signal_ready()
    logger.info("Système prêt — en attente de présence...")
    logger.info("-" * 55)

    try:
        while True:
            _process_one_detection(cam, anti_spoof, extractor)
            time.sleep(config.DEBOUNCE_DELAY)

    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur (Ctrl+C)")

    finally:
        cam.close()
        gpio_feedback.cleanup()
        logger.info("Système arrêté proprement")


def _process_one_detection(
    cam: CameraManager,
    anti_spoof: AntiSpoofPredict,
    extractor: EmbeddingExtractor,
):
    # ── ÉTAPE 1 : Attendre PIR ───────────────────────────────
    cam.set_status("En attente de présence...", COLOR_WHITE)
    cam.set_bbox(None)
    logger.info("En attente de présence (PIR)...")

    if not pir.wait_for_motion():
        return

    logger.info("Présence détectée !")
    cam.set_status("Présence détectée — analyse en cours...", COLOR_ORANGE)
    gpio_feedback.signal_processing()

    # ── ÉTAPE 2 : Capture image ──────────────────────────────
    try:
        image_bytes = cam.capture_image()
    except Exception as e:
        logger.error(f"Erreur capture image : {e}")
        cam.set_status("⚠ Erreur caméra", COLOR_RED)
        gpio_feedback.signal_error()
        return

    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Image invalide après décodage")
        cam.set_status("⚠ Image invalide", COLOR_RED)
        gpio_feedback.signal_spoof_detected()
        return

    # ── ÉTAPE 3 : Anti-spoof ─────────────────────────────────
    cam.set_status("Vérification de vivacité...", COLOR_ORANGE)
    logger.info("Anti-spoof prediction en cours...")

    model_path = _find_model()
    if model_path is None:
        logger.error("Aucun modèle .pth trouvé dans ./models")
        cam.set_status("⚠ Modèle anti-spoof manquant", COLOR_RED)
        gpio_feedback.signal_error()
        return

    try:
        scores     = anti_spoof.predict(img, model_path)
        fake_score = float(scores[0][0])
        real_score = float(scores[0][1])
        logger.info(f"Anti-spoof — Fake: {fake_score:.4f} | Real: {real_score:.4f}")

        if real_score < 0.5:
            logger.warning(f"SPOOF détecté (score: {real_score:.4f})")
            cam.set_status(f"✗ Tentative de fraude détectée ({real_score:.2f})", COLOR_RED)
            gpio_feedback.signal_spoof_detected()
            time.sleep(2)
            return

        logger.info(f"Liveness validée (score: {real_score:.4f})")
        cam.set_status(f"Vivacité confirmée ({real_score:.2f}) — identification...", COLOR_ORANGE)

    except Exception as e:
        logger.error(f"Erreur anti-spoof : {e}")
        cam.set_status("⚠ Erreur anti-spoof", COLOR_RED)
        gpio_feedback.signal_error()
        return

    # ── ÉTAPE 4 : Extraction embedding InsightFace ───────────
    cam.set_status("Extraction biométrique en cours...", COLOR_ORANGE)
    logger.info("Extraction embedding facial (InsightFace)...")

    embedding = extractor.extract_to_list(img)
    if embedding is None:
        logger.warning("Aucun visage détecté par InsightFace")
        cam.set_status("✗ Aucun visage détecté — accès refusé", COLOR_RED)
        gpio_feedback.signal_access_denied()
        time.sleep(2)
        return

    logger.info(f"Embedding extrait — {len(embedding)} valeurs")
    cam.set_status("Identification en cours...", COLOR_ORANGE)

    # ── ÉTAPE 5 : Envoi embedding au serveur Django ──────────
    logger.info("Envoi embedding au serveur Django...")
    result = api_client.send_embedding(embedding)

    # ── ÉTAPE 6 : Décision ───────────────────────────────────
    if not result.get("success"):
        error = result.get("error", "Erreur inconnue")
        logger.warning(f"API erreur : {error}")
        cam.set_status(f"✗ Accès refusé — {error[:40]}", COLOR_RED)

        if "inaccessible" in error or "Timeout" in error:
            gpio_feedback.signal_error()
        else:
            gpio_feedback.signal_access_denied()

        time.sleep(2)
        return

    # ── ÉTAPE 7 : Accès accordé ───────────────────────────────
    user     = result.get("user", {})
    distance = result.get("distance", "N/A")
    name     = user.get("name", "Inconnu") if isinstance(user, dict) else str(user)

    logger.info(f"Utilisateur reconnu : {user} | Distance : {distance}")
    cam.set_status(f"✓ Accès autorisé — Bienvenue {name} !", COLOR_GREEN)
    gpio_feedback.signal_access_granted()

    time.sleep(3)
    logger.info("-" * 55)


def _find_model() -> str | None:
    models_dir = "./models"
    if not os.path.isdir(models_dir):
        return None
    for f in os.listdir(models_dir):
        if f.endswith(".pth"):
            return os.path.join(models_dir, f)
    return None


if __name__ == "__main__":
    main()