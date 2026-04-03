# ============================================================
# src/liveness.py
# Liveness detection ultra-optimisée pour Raspberry Pi 5
# Méthode : multi-check (clignement + texture + saturation)
# Compatible PiCamera2 / OpenCV
# ============================================================

import cv2
import numpy as np
import time
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import config

logger = logging.getLogger(__name__)

# ── Détecteurs Haar Cascade intégrés OpenCV
def _load_detectors():
    """Charge les cascades Haar pour visage et yeux"""
    opencv_data = cv2.data.haarcascades
    face_cascade = cv2.CascadeClassifier(
        os.path.join(opencv_data, "haarcascade_frontalface_default.xml")
    )
    eye_cascade = cv2.CascadeClassifier(
        os.path.join(opencv_data, "haarcascade_eye.xml")
    )
    if face_cascade.empty() or eye_cascade.empty():
        raise RuntimeError("Impossible de charger les cascades Haar")
    return face_cascade, eye_cascade

# ── Calcul Eye Aspect Ratio (EAR)
def compute_ear(eye_points: np.ndarray) -> float:
    """Calcule l'Eye Aspect Ratio à partir de 6 landmarks"""
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    return round((A + B) / (2.0 * C) if C != 0 else 0.0, 4)

# ── Texture score
def _compute_texture_score(gray_face: np.ndarray) -> float:
    """Score texture via variance du Laplacien"""
    return float(cv2.Laplacian(gray_face, cv2.CV_64F).var())

# ── Saturation score
def _compute_saturation_score(color_face: np.ndarray) -> float:
    """Score de saturation HSV"""
    hsv = cv2.cvtColor(color_face, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())

# ── Liveness sur image statique
def check_liveness_opencv(image_bytes: bytes) -> dict:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"is_live": False, "reason": "Image invalide", "face_detected": False,
                "eyes_detected": 0, "texture_score": 0}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade, eye_cascade = _load_detectors()

    # Redimension si trop grand
    h, w = gray.shape
    if w > 640:
        scale = 640 / w
        gray = cv2.resize(gray, (640, int(h*scale)))
        img = cv2.resize(img, (640, int(h*scale)))

    # Détecter visage
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    if len(faces) == 0:
        return {"is_live": False, "reason": "Aucun visage détecté", "face_detected": False,
                "eyes_detected": 0, "texture_score": 0}

    fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
    face_gray = gray[fy:fy+fh, fx:fx+fw]
    face_color = img[fy:fy+fh, fx:fx+fw]

    # Détecter yeux
    eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5, minSize=(20, 20))
    eyes_count = len(eyes)

    # Score texture et saturation
    texture_score = _compute_texture_score(face_gray)
    sat_score = _compute_saturation_score(face_color)

    # Règles strictes anti-spoofing
    if eyes_count < 1:
        return {"is_live": False, "reason": "Aucun oeil détecté", "face_detected": True,
                "eyes_detected": eyes_count, "texture_score": texture_score}
    if texture_score < 15:
        return {"is_live": False, "reason": "Texture trop uniforme", "face_detected": True,
                "eyes_detected": eyes_count, "texture_score": texture_score}
    if sat_score > 120:
        return {"is_live": False, "reason": "Saturation anormale", "face_detected": True,
                "eyes_detected": eyes_count, "texture_score": texture_score}

    return {"is_live": True, "reason": "Visage vivant détecté", "face_detected": True,
            "eyes_detected": eyes_count, "texture_score": texture_score}

# ── Liveness temps réel (PiCamera2 ou webcam)
def check_liveness_realtime(camera) -> dict:
    face_cascade, eye_cascade = _load_detectors()
    blink_count = 0
    eyes_closed = False
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > config.LIVENESS_TIMEOUT:
            return {"is_live": False, "reason": "Timeout liveness", "blink_count": blink_count}

        ret, frame = camera.read()
        if not ret:
            return {"is_live": False, "reason": "Erreur lecture caméra", "blink_count": blink_count}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
        if len(faces) == 0:
            continue

        fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        face_gray = gray[fy:fy+fh, fx:fx+fw]
        face_color = frame[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5, minSize=(20,20))
        eyes_count = len(eyes)

        # Détection clignement
        if eyes_count >= 2:
            if eyes_closed:
                blink_count += 1
                eyes_closed = False
                if blink_count >= config.BLINK_COUNT_REQUIRED:
                    return {"is_live": True,
                            "reason": f"{blink_count} clignement(s) détecté(s)",
                            "blink_count": blink_count}
        else:
            eyes_closed = True

        time.sleep(0.05)