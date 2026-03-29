# ============================================================
# src/liveness.py
# Rôle : détecter si le visage est celui d'une personne
#        vivante (et non une photo ou vidéo)
#
# Méthode : Eye Aspect Ratio (EAR)
#   - Analyse les landmarks des yeux en temps réel
#   - Détecte le clignement naturel des yeux
#   - Une photo ne peut pas cligner → spoof détecté
#
# Utilisé par : main.py
# Dépend de   : config.py, gpio_feedback.py
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


# ── Chargement du détecteur de visage OpenCV ────────────────
# On utilise le détecteur Haar Cascade d'OpenCV
# Plus léger que dlib — parfait pour le Raspberry Pi
def _load_detectors():
    """
    Charge les détecteurs OpenCV pour visage et yeux.
    Retourne (face_cascade, eye_cascade)
    """
    # Chemin vers les fichiers Haar Cascade d'OpenCV
    opencv_data = cv2.data.haarcascades

    face_cascade = cv2.CascadeClassifier(
        os.path.join(opencv_data, 'haarcascade_frontalface_default.xml')
    )
    eye_cascade = cv2.CascadeClassifier(
        os.path.join(opencv_data, 'haarcascade_eye.xml')
    )

    if face_cascade.empty():
        raise RuntimeError("Impossible de charger haarcascade_frontalface_default.xml")
    if eye_cascade.empty():
        raise RuntimeError("Impossible de charger haarcascade_eye.xml")

    logger.info("Détecteurs OpenCV chargés avec succès")
    return face_cascade, eye_cascade


# ── Calcul du EAR (Eye Aspect Ratio) ────────────────────────
def compute_ear(eye_points: np.ndarray) -> float:
    """
    Calcule l'Eye Aspect Ratio à partir de 6 points de l'oeil.

    Formule :
        EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)

    Les 6 points sont :
        p1 = coin gauche de l'oeil
        p2 = haut gauche
        p3 = haut droit
        p4 = coin droit
        p5 = bas droit
        p6 = bas gauche

    Valeurs typiques :
        Oeil ouvert  → EAR ≈ 0.25 à 0.35
        Oeil fermé   → EAR ≈ 0.10 à 0.20
        Clignement   → EAR descend sous EAR_THRESHOLD (0.25)

    Paramètre :
        eye_points : numpy array de shape (6, 2)

    Retourne :
        float — valeur EAR entre 0 et 1
    """
    # Distance verticale haute gauche → bas gauche
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    # Distance verticale haute droite → bas droite
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    # Distance horizontale coin gauche → coin droit
    C = np.linalg.norm(eye_points[0] - eye_points[3])

    # Éviter division par zéro
    if C == 0:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return round(float(ear), 4)


# ── Détection liveness avec OpenCV ──────────────────────────
def check_liveness_opencv(image_bytes: bytes) -> dict:
    """
    Version simplifiée de liveness detection avec OpenCV.
    Analyse UNE image statique pour détecter :
        1. Présence d'un visage
        2. Présence des deux yeux
        3. Score de texture (peau vs papier/écran)

    Utilisé quand on n'a pas accès au flux vidéo temps réel.

    Paramètre :
        image_bytes : bytes de l'image capturée

    Retourne :
    {
        "is_live"      : True / False
        "reason"       : explication de la décision
        "face_detected": True / False
        "eyes_detected": nombre d'yeux détectés
        "texture_score": score de texture (> 50 = naturel)
    }
    """
    # ── 1. Décoder l'image ───────────────────────────────────
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "is_live":       False,
            "reason":        "Image invalide",
            "face_detected": False,
            "eyes_detected": 0,
            "texture_score": 0
        }

    # ── 2. Convertir en niveaux de gris ──────────────────────
    # Les détecteurs Haar travaillent sur des images grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── 3. Charger les détecteurs ────────────────────────────
    try:
        face_cascade, eye_cascade = _load_detectors()
    except RuntimeError as e:
        logger.error(f"Erreur chargement détecteurs : {e}")
        return {
            "is_live":       False,
            "reason":        f"Erreur système : {e}",
            "face_detected": False,
            "eyes_detected": 0,
            "texture_score": 0
        }

  # ── 4. Détecter le visage ────────────────────────────────
    # Redimensionner d'abord si l'image est trop grande
    h, w = gray.shape
    if w > 640:
        scale      = 640 / w
        new_w      = 640
        new_h      = int(h * scale)
        gray       = cv2.resize(gray, (new_w, new_h))
        img        = cv2.resize(img,  (new_w, new_h))
        logger.info(f"Image redimensionnée : {w}x{h} → {new_w}x{new_h}")

    # Essayer plusieurs configurations
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80)
    )

    if len(faces) == 0:
        logger.info("Tentative 2 — paramètres assouplis")
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(30,30)
        )

    if len(faces) == 0:
        logger.info("Liveness : aucun visage détecté")
        return {
            "is_live":       False,
            "reason":        "Aucun visage détecté",
            "face_detected": False,
            "eyes_detected": 0,
            "texture_score": 0
        }

  # ── 4. Détecter le visage ────────────────────────────────
    # Redimensionner d'abord si l'image est trop grande
    h, w = gray.shape
    if w > 640:
        scale      = 640 / w
        new_w      = 640
        new_h      = int(h * scale)
        gray       = cv2.resize(gray, (new_w, new_h))
        img        = cv2.resize(img,  (new_w, new_h))
        logger.info(f"Image redimensionnée : {w}x{h} → {new_w}x{new_h}")

    # Essayer plusieurs configurations
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80)
    )

    if len(faces) == 0:
        logger.info("Tentative 2 — paramètres assouplis")
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(30,30)
        )

    if len(faces) == 0:
        logger.info("Liveness : aucun visage détecté")
        return {
            "is_live":       False,
            "reason":        "Aucun visage détecté",
            "face_detected": False,
            "eyes_detected": 0,
            "texture_score": 0
        }

    # Prendre le plus grand visage détecté
    face            = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh  = face
    face_gray       = gray[fy:fy+fh, fx:fx+fw]
    face_color      = img[fy:fy+fh, fx:fx+fw]

    # ── 5. Détecter les yeux dans le visage ──────────────────
    eyes = eye_cascade.detectMultiScale(
        face_gray,
        scaleFactor = 1.1,
        minNeighbors= 5,
        minSize     = (20, 20)
    )

    eyes_count = len(eyes)
    logger.info(f"Liveness : {eyes_count} oeil(s) détecté(s)")

    # ── 6. Analyse de texture ────────────────────────────────
    # La peau humaine a une texture naturelle variée
    # Une photo imprimée ou un écran a une texture uniforme
    texture_score = _compute_texture_score(face_gray)
    logger.info(f"Liveness : texture score = {texture_score:.1f}")

    # ── 7. Analyse saturation ────────────────────────────────
    # La peau vivante a une saturation naturelle
    # Un écran a une saturation très élevée
    sat_score = _compute_saturation_score(face_color)
    logger.info(f"Liveness : saturation score = {sat_score:.1f}")

    # ── 8. Décision finale ───────────────────────────────────
    # Règles de décision :
    #   - Au moins 1 oeil détecté
    #   - Texture suffisamment variée (> 15)
    #   - Saturation dans la plage naturelle (< 120)

    if eyes_count == 0:
        return {
            "is_live":       False,
            "reason":        "Aucun oeil détecté — possible photo sans yeux visibles",
            "face_detected": True,
            "eyes_detected": 0,
            "texture_score": texture_score
        }

    if texture_score < 15:
        return {
            "is_live":       False,
            "reason":        f"Texture trop uniforme ({texture_score:.1f}) — possible photo imprimée",
            "face_detected": True,
            "eyes_detected": eyes_count,
            "texture_score": texture_score
        }

    if sat_score > 120:
        return {
            "is_live":       False,
            "reason":        f"Saturation anormale ({sat_score:.1f}) — possible écran",
            "face_detected": True,
            "eyes_detected": eyes_count,
            "texture_score": texture_score
        }

    # Tous les checks passés → vivant
    return {
        "is_live":       True,
        "reason":        "Visage vivant détecté",
        "face_detected": True,
        "eyes_detected": eyes_count,
        "texture_score": texture_score
    }


# ── Liveness temps réel (flux vidéo) ────────────────────────
def check_liveness_realtime(camera) -> dict:
    """
    Liveness detection en temps réel sur flux vidéo.
    Demande à la personne de cligner des yeux.

    Utilisé sur le vrai Raspberry Pi avec PiCamera2.

    Paramètre :
        camera : objet caméra OpenCV (cv2.VideoCapture)

    Retourne :
    {
        "is_live"     : True / False
        "reason"      : explication
        "blink_count" : nombre de clignements détectés
    }
    """
    logger.info(f"Liveness temps réel — attend {config.BLINK_COUNT_REQUIRED} clignement(s)")

    face_cascade, eye_cascade = _load_detectors()

    blink_count   = 0
    eyes_closed   = False
    start_time    = time.time()

    while True:
        # Vérifier le timeout
        elapsed = time.time() - start_time
        if elapsed > config.LIVENESS_TIMEOUT:
            logger.warning(f"Timeout liveness après {elapsed:.1f}s")
            return {
                "is_live":     False,
                "reason":      f"Timeout — aucun clignement en {config.LIVENESS_TIMEOUT}s",
                "blink_count": blink_count
            }

        # Lire une frame
        ret, frame = camera.read()
        if not ret:
            logger.error("Impossible de lire la caméra")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détecter visage
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80,80))
        if len(faces) == 0:
            continue

        # Prendre le plus grand visage
        fx, fy, fw, fh = max(faces, key=lambda f: f[2]*f[3])
        face_gray = gray[fy:fy+fh, fx:fx+fw]

        # Détecter yeux
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5, minSize=(20,20))
        eyes_count = len(eyes)

        # Logique clignement :
        # yeux ouverts (≥ 2) → yeux fermés (0-1) → yeux ouverts = 1 clignement
        if eyes_count >= 2:
            if eyes_closed:
                # Les yeux se rouvrent → clignement complet !
                blink_count += 1
                eyes_closed  = False
                logger.info(f"Clignement détecté ! Total : {blink_count}")

                if blink_count >= config.BLINK_COUNT_REQUIRED:
                    logger.info("Liveness validée par clignement !")
                    return {
                        "is_live":     True,
                        "reason":      f"{blink_count} clignement(s) détecté(s)",
                        "blink_count": blink_count
                    }
        else:
            # Yeux fermés ou non détectés
            eyes_closed = True

        time.sleep(0.05)  # 20 fps

    return {
        "is_live":     False,
        "reason":      "Erreur lecture caméra",
        "blink_count": blink_count
    }


# ── Fonctions utilitaires ────────────────────────────────────
def _compute_texture_score(gray_face: np.ndarray) -> float:
    """
    Calcule un score de texture via le Laplacien.
    Mesure la variance des gradients dans l'image.

    Score élevé  → texture naturelle (peau vivante)
    Score faible → texture uniforme (photo imprimée)
    """
    laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
    return float(laplacian.var())


def _compute_saturation_score(color_face: np.ndarray) -> float:
    """
    Calcule la saturation moyenne du visage en HSV.

    Score faible  → peau naturelle (10-80)
    Score élevé   → écran ou impression couleur (> 100)
    """
    hsv = cv2.cvtColor(color_face, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())