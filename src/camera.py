import cv2
from picamera2 import Picamera2
import time
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import config

logger = logging.getLogger(__name__)
picamera2 = None


def open_camera():
    
    global picamera2
    picamera2 = Picamera2()
    cam_config = picamera2.create_preview_configuration(
        main={"format": "RGB888", "size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)}
    )
    picamera2.configure(cam_config)
    picamera2.start()
    time.sleep(config.CAMERA_WARMUP)
    logger.info("Caméra démarrée (%dx%d)", config.CAMERA_WIDTH, config.CAMERA_HEIGHT)


def close_camera():
    
    global picamera2
    if picamera2 is not None:
        picamera2.stop()
        picamera2 = None
        logger.info("Caméra arrêtée.")


def capture_image_opencv():
   
    if picamera2 is None:
        raise RuntimeError("Caméra non initialisée — appeler open_camera() d'abord")

    image_rgb = picamera2.capture_array()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    success, buffer = cv2.imencode(".jpg", image_bgr)
    if not success:
        raise RuntimeError("Échec de l'encodage JPEG")

    logger.debug("Image capturée (%dx%d)", image_bgr.shape[1], image_bgr.shape[0])
    return buffer.tobytes()


def get_video_stream():
    """Retourne l'instance Picamera2 pour le flux vidéo (liveness)."""
    return picamera2


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    open_camera()
    try:
        img_bytes = capture_image_opencv()
        with open("capture_test.jpg", "wb") as f:
            f.write(img_bytes)
        logger.info("Image enregistrée sous capture_test.jpg (%d octets)", len(img_bytes))
    finally:
        close_camera()