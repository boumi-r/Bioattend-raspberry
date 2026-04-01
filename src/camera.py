import cv2
import time
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import config

logger = logging.getLogger(__name__)


try:
    from picamera2 import Picamera2
    IS_RASPBERRY = True
    logger.info("PiCamera2 disponible — mode Raspberry Pi réel")
except ImportError:
    IS_RASPBERRY = False
    logger.warning("PiCamera2 non disponible — mode webcam OpenCV")



class CameraManager:
  
    def __init__(self):
        self.camera     = None
        self.is_open    = False
        self.camera_index = config.PICAMERA_INDEX

    def open(self):
       
        if IS_RASPBERRY:
            self._open_picamera()
        else:
            self._open_opencv()

    def _open_picamera(self):
        """Démarre PiCamera2 sur le vrai Raspberry Pi."""
        try:
            camera_info = []
            if hasattr(Picamera2, "global_camera_info"):
                camera_info = Picamera2.global_camera_info() or []

            if camera_info:
                if self.camera_index >= len(camera_info):
                    raise RuntimeError(
                        f"PICAMERA_INDEX={self.camera_index} invalide (caméras détectées: {len(camera_info)})."
                    )
                logger.info(f"Caméras détectées: {len(camera_info)} — utilisation index {self.camera_index}")
            else:
                logger.warning("Aucune caméra listée par libcamera, tentative d'ouverture quand même...")

            self.camera = Picamera2(self.camera_index)
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

        except IndexError:
            raise RuntimeError(
                "Aucune caméra accessible via PiCamera2 (IndexError). Vérifie le branchement nappe/capteur, "
                "active la caméra (raspi-config), puis teste avec 'libcamera-hello'."
            )
        except Exception as e:
            logger.error(f"Erreur ouverture PiCamera2 : {e}")
            raise

    def _open_opencv(self):
        """
        Démarre la caméra OpenCV sur Codespaces.
        Nécessite une webcam accessible.
        """
        self.camera = cv2.VideoCapture(0)

        if self.camera.isOpened():
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.is_open = True
            logger.info("Webcam OpenCV ouverte")
        else:
            # Pas de webcam — arrêt explicite
            self.camera = None
            self.is_open = False
            raise RuntimeError("Aucune webcam détectée (OpenCV index 0 indisponible).")

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
        """Capture avec OpenCV."""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if not ret:
                raise RuntimeError("Impossible de lire la webcam.")

            success, buffer = cv2.imencode(
                '.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            logger.info(f"Image webcam capturée : {len(buffer.tobytes())} bytes")
            return buffer.tobytes()

        raise RuntimeError("Flux webcam non initialisé.")

    def get_video_stream(self):
        """
        Retourne un objet compatible cv2.VideoCapture
        pour le liveness temps réel.

        Sur Pi    → crée un VideoCapture depuis PiCamera2
        Sur autres → retourne la webcam OpenCV directement
        """
        if IS_RASPBERRY:
            # Sur Pi on utilise le flux PiCamera2 via OpenCV
            return cv2.VideoCapture(self.camera_index)
        else:
            if self.camera and self.camera.isOpened():
                return self.camera
            else:
                logger.warning("Pas de flux vidéo disponible")
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