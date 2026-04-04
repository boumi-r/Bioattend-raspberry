# ============================================================
# src/camera.py
# Gestion caméra avec PiCamera2 (Raspberry Pi)
# ============================================================

from picamera2 import Picamera2
import cv2
import time
import logging
import config

logger = logging.getLogger(__name__)


class CameraManager:
    def __init__(self):
        self.camera = None
        self.is_open = False

    def open(self):
        """Initialise et démarre la caméra PiCamera2"""
        try:
            self.camera = Picamera2()

            config_cam = self.camera.create_preview_configuration(
                main={
                    "size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
                    "format": "RGB888"
                }
            )

            self.camera.configure(config_cam)
            self.camera.start()

            logger.info(f"Caméra démarrée — chauffe {config.CAMERA_WARMUP}s...")
            time.sleep(config.CAMERA_WARMUP)

            self.is_open = True
            logger.info("Caméra prête")

        except Exception as e:
            logger.error(f"Erreur caméra : {e}")
            raise

    # --------------------------------------------------------
    # 🔹 Capture frame pour traitement IA (OpenCV)
    # --------------------------------------------------------
    def capture_frame(self):
        """
        Capture une image (numpy array BGR)
        Utilisé pour liveness et traitement OpenCV
        """
        if not self.is_open:
            raise RuntimeError("Caméra non ouverte")

        try:
            frame = self.camera.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame_bgr

        except Exception as e:
            logger.error(f"Erreur capture frame : {e}")
            raise

    # --------------------------------------------------------
    # 🔹 Capture JPEG pour API
    # --------------------------------------------------------
    def capture_jpeg(self):
        """
        Capture une image et retourne bytes JPEG
        Utilisé pour envoi API Django
        """
        try:
            frame = self.capture_frame()

            success, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

            if not success:
                raise RuntimeError("Erreur encodage JPEG")

            image_bytes = buffer.tobytes()
            logger.info(f"Image capturée : {len(image_bytes)} bytes")

            return image_bytes

        except Exception as e:
            logger.error(f"Erreur capture JPEG : {e}")
            raise

    # --------------------------------------------------------
    # 🔹 Capture multiple frames (optionnel pour liveness avancé)
    # --------------------------------------------------------
    def capture_frames(self, num_frames=5, delay=0.1):
        """
        Capture plusieurs frames (utile pour blink detection)
        """
        frames = []

        for _ in range(num_frames):
            try:
                frame = self.capture_frame()
                frames.append(frame)
                time.sleep(delay)
            except Exception as e:
                logger.warning(f"Erreur frame multiple : {e}")

        return frames

    # --------------------------------------------------------
    # 🔹 Fermeture propre
    # --------------------------------------------------------
    def close(self):
        """Ferme proprement la caméra"""
        try:
            if self.camera:
                self.camera.stop()
                self.camera.close()
                logger.info("Caméra fermée")

        except Exception as e:
            logger.error(f"Erreur fermeture caméra : {e}")

        finally:
            self.is_open = False