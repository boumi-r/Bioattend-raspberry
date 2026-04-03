# src/camera.py

from picamera2 import Picamera2
import cv2
import time
import logging
import config

logger = logging.getLogger(__name__)


class CameraManager:
    def __init__(self):
        self.camera = None

    def open(self):
        """Initialise la caméra PiCamera2"""
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

            logger.info("Caméra prête")

        except Exception as e:
            logger.error(f"Erreur caméra : {e}")
            raise

    def capture_frame(self):
        """
        Capture une image (numpy array BGR pour OpenCV)
        """
        frame = self.camera.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def capture_jpeg(self):
        """
        Capture une image et retourne bytes JPEG
        """
        frame = self.capture_frame()

        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise RuntimeError("Erreur encodage JPEG")

        return buffer.tobytes()

    def close(self):
        """Ferme la caméra"""
        if self.camera:
            self.camera.stop()
            self.camera.close()
            logger.info("Caméra fermée")