# ============================================================
# src/liveness.py (VERSION CORRIGÉE)
# ============================================================

import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

# ── Charger UNE SEULE FOIS les détecteurs ───────────────────
opencv_data = cv2.data.haarcascades

face_cascade = cv2.CascadeClassifier(
    os.path.join(opencv_data, 'haarcascade_frontalface_default.xml')
)
eye_cascade = cv2.CascadeClassifier(
    os.path.join(opencv_data, 'haarcascade_eye.xml')
)

if face_cascade.empty() or eye_cascade.empty():
    raise RuntimeError("Erreur chargement Haar Cascade")

logger.info("Détecteurs OpenCV chargés")


# ============================================================
# 🎯 LIVENESS PRINCIPAL (FRAME)
# ============================================================
def check_liveness_opencv(frame) -> dict:
    """
    Analyse UNE image (numpy array BGR)

    Retour :
    {
        "is_live": True/False,
        "reason": "...",
        "face_detected": bool,
        "eyes_detected": int,
        "texture_score": float
    }
    """

    if frame is None:
        return _fail("Image invalide")

    # ── 1. Convertir en gris ─────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── 2. Resize pour perf ──────────────────────────
    h, w = gray.shape
    if w > 640:
        scale = 640 / w
        gray  = cv2.resize(gray, (640, int(h * scale)))
        frame = cv2.resize(frame, (640, int(h * scale)))

    # ── 3. Détection visage ─────────────────────────
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) == 0:
        return _fail("Aucun visage détecté")

    # Prendre le plus grand visage
    fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
    face_gray  = gray[fy:fy+fh, fx:fx+fw]
    face_color = frame[fy:fy+fh, fx:fx+fw]

    # ── 4. Détection yeux ───────────────────────────
    eyes = eye_cascade.detectMultiScale(
        face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )

    eyes_count = len(eyes)
    logger.info(f"Eyes détectés : {eyes_count}")

    # ── 5. Texture ──────────────────────────────────
    texture_score = _compute_texture_score(face_gray)

    # ── 6. Saturation ───────────────────────────────
    sat_score = _compute_saturation_score(face_color)

    logger.info(f"Texture : {texture_score:.1f} | Saturation : {sat_score:.1f}")

    # ── 7. Décision ─────────────────────────────────
    if eyes_count == 0:
        return _fail("Aucun oeil détecté", True, eyes_count, texture_score)

    if texture_score < 15:
        return _fail("Texture suspecte (photo)", True, eyes_count, texture_score)

    if sat_score > 120:
        return _fail("Saturation élevée (écran)", True, eyes_count, texture_score)

    return {
        "is_live": True,
        "reason": "Visage vivant détecté",
        "face_detected": True,
        "eyes_detected": eyes_count,
        "texture_score": texture_score
    }


# ============================================================
# 🔧 UTILS
# ============================================================
def _fail(reason, face=False, eyes=0, texture=0):
    return {
        "is_live": False,
        "reason": reason,
        "face_detected": face,
        "eyes_detected": eyes,
        "texture_score": texture
    }


def _compute_texture_score(gray_face):
    laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
    return float(laplacian.var())


def _compute_saturation_score(color_face):
    hsv = cv2.cvtColor(color_face, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())