# ============================================================
# src/camera.py
# Rôle : capturer une image avec PiCamera2
#
# Sur Raspberry Pi  → utilise PiCamera2 (vraie caméra)
# Sur Codespaces    → utilise OpenCV webcam ou image simulée
#
# Utilisé par : main.py
# Dépend de   : config.py
# ============================================================
import cv2
import numpy as np
import time
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import config

logger = logging.getLogger(__name__)


# ── Chargement PiCamera2 (réel ou simulé) ───────────────────
try:
    from picamera2 import Picamera2
    IS_RASPBERRY = True
    logger.info("PiCamera2 disponible — mode Raspberry Pi réel")
except ImportError:
    IS_RASPBERRY = False
    logger.warning("PiCamera2 non disponible — mode simulation caméra")


# ── Classe CameraManager ─────────────────────────────────────
class CameraManager:
    """
    Gère la caméra selon l'environnement :
    - Raspberry Pi → PiCamera2
    - Codespaces   → OpenCV (webcam ou image simulée)
    """

    def __init__(self):
        self.camera     = None
        self.is_open    = False

    def open(self):
        """
        Initialise et démarre la caméra.
        Doit être appelée une seule fois au démarrage.
        """
        if IS_RASPBERRY:
            self._open_picamera()
        else:
            self._open_opencv()

    def _open_picamera(self):
        """Démarre PiCamera2 sur le vrai Raspberry Pi."""
        try:
            self.camera = Picamera2()
            config_cam  = self.camera.create_still_configuration(
                main={
                    "size":   (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
                    "format": "RGB888"
                }
            )
            self.camera.configure(config_cam)
            self.camera.start()

            # Temps de chauffe — laisse le capteur s'ajuster à la lumière
            logger.info(f"Caméra PiCamera2 démarrée — chauffe {config.CAMERA_WARMUP}s...")
            time.sleep(config.CAMERA_WARMUP)

            self.is_open = True
            logger.info("PiCamera2 prête")

        except Exception as e:
            logger.error(f"Erreur ouverture PiCamera2 : {e}")
            raise

    def _open_opencv(self):
        """
        Démarre la caméra OpenCV sur Codespaces.
        Essaie d'ouvrir une webcam, sinon mode simulation.
        """
        self.camera = cv2.VideoCapture(0)

        if self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.is_open = True
            logger.info("Webcam OpenCV ouverte")
        else:
            # Pas de webcam — mode simulation pure
            self.camera = None
            self.is_open = True
            logger.warning("Aucune webcam — mode simulation image activé")

    def capture_image(self) -> bytes:
        """
        Capture une image et retourne les bytes JPEG.

        Retourne :
            bytes — image JPEG prête à envoyer à l'API Django
        """
        if IS_RASPBERRY:
            return self._capture_picamera()
        else:
            return self._capture_opencv()

    def _capture_picamera(self) -> bytes:
        """Capture avec PiCamera2."""
        try:
            # Capturer en numpy array RGB
            frame = self.camera.capture_array()

            # Convertir RGB → BGR pour OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Encoder en JPEG
            success, buffer = cv2.imencode(
                '.jpg', frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            if not success:
                raise RuntimeError("Échec encodage JPEG")

            logger.info(f"Image capturée : {len(buffer.tobytes())} bytes")
            return buffer.tobytes()

        except Exception as e:
            logger.error(f"Erreur capture PiCamera2 : {e}")
            raise

    def _capture_opencv(self) -> bytes:
        """Capture avec OpenCV ou génère une image simulée."""
        if self.camera and self.camera.isOpened():
            # Vraie webcam disponible
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Impossible de lire la webcam")
                return self._generate_test_image()

            success, buffer = cv2.imencode(
                '.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            logger.info(f"Image webcam capturée : {len(buffer.tobytes())} bytes")
            return buffer.tobytes()

        else:
            # Mode simulation — retourner image de test si disponible
            return self._generate_test_image()

    def _generate_test_image(self) -> bytes:
        """
        Génère une image de test en simulation.
        Cherche RAOUL.jpg ou crée une image grise.
        """
        # Chercher une image de test
        for name in ["RAOUL.jpg", "test_face.jpg", "face.jpg"]:
            if os.path.exists(name):
                with open(name, "rb") as f:
                    logger.info(f"[SIMULATION] Image de test : {name}")
                    return f.read()

        # Sinon créer une image grise basique
        logger.warning("[SIMULATION] Génération image grise")
        img     = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:]  = (128, 128, 128)
        _, buf  = cv2.imencode('.jpg', img)
        return buf.tobytes()

    def get_video_stream(self):
        """
        Retourne un objet compatible cv2.VideoCapture
        pour le liveness temps réel.

        Sur Pi    → crée un VideoCapture depuis PiCamera2
        Sur autres → retourne la webcam OpenCV directement
        """
        if IS_RASPBERRY:
            # Sur Pi on utilise le flux PiCamera2 via OpenCV
            return cv2.VideoCapture(0)
        else:
            if self.camera and self.camera.isOpened():
                return self.camera
            else:
                logger.warning("Pas de flux vidéo disponible — simulation")
                return None

    def close(self):
        """Ferme proprement la caméra."""
        if IS_RASPBERRY and self.camera:
            self.camera.stop()
            self.camera.close()
            logger.info("PiCamera2 fermée")
        elif self.camera and hasattr(self.camera, 'release'):
            self.camera.release()
            logger.info("Caméra OpenCV fermée")
        self.is_open = False