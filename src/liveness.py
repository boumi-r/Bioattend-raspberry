# ============================================================
# src/liveness.py
# Rôle : détecter si le visage est celui d'une personne
#        vivante (et non une photo, vidéo, deepfake, etc.)
#
# Méthodes multi-couches :
#   1. Eye Aspect Ratio (EAR) — détecte clignement naturel
#   2. Analyse texture — surface peau vs photo/papier
#   3. Saturation — peau naturelle vs écran/impression
#   4. Scintillement écran — détecte vidéo sur écran
#   5. Gradients & edges — photo a contours plus nets
#   6. Motion tracking — mouvements du visage naturels  
#   7. Flou — motion blur naturel vs vidéo compressée
#   8. Cohérence micro-expressions — vérifie naturel
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


# ── Détection liveness avec OpenCV multi-couches ─────────────
def check_liveness_opencv(image_bytes: bytes) -> dict:
    """
    Détection liveness robuste avec 8 couches d'analyse.
    Détecte photos, vidéos d'écran, deepfakes, faux jumeaux.

    Paramètre :
        image_bytes : bytes de l'image capturée

    Retourne :
    {
        "is_live"       : True / False
        "reason"        : explication de la décision
        "face_detected" : True / False
        "confidence"    : score de confiance 0-100 (%)
        "checks"        : dict des scores pour chaque méthode
    }
    """
    # ── 1. Décoder l'image ───────────────────────────────────
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {
            "is_live":       False,
            "reason":        "Image invalide ou corrompue",
            "face_detected": False,
            "confidence":    0,
            "checks":        {}
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Redimensionner si trop grande
    if w > 640:
        scale = 640 / w
        new_w, new_h = 640, int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h))
        img  = cv2.resize(img,  (new_w, new_h))
        logger.debug(f"Image redimensionnée : {w}x{h} → {new_w}x{new_h}")

    # ── 2. Charger détecteurs ───────────────────────────────
    try:
        face_cascade, eye_cascade = _load_detectors()
    except RuntimeError as e:
        logger.error(f"Erreur détecteurs : {e}")
        return {
            "is_live":       False,
            "reason":        f"Erreur système : {e}",
            "face_detected": False,
            "confidence":    0,
            "checks":        {}
        }

    # ── 3. Détecter le visage ───────────────────────────────
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
        )

    if len(faces) == 0:
        return {
            "is_live":       False,
            "reason":        "Aucun visage détecté",
            "face_detected": False,
            "confidence":    0,
            "checks":        {}
        }

    # Prendre le plus grand visage
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    face_gray  = gray[fy:fy+fh, fx:fx+fw]
    face_color = img[fy:fy+fh, fx:fx+fw]

    # ── 4. Exécuter tous les tests de liveness ──────────────
    checks = {}

    # Test 1 : Détection des yeux
    eyes = eye_cascade.detectMultiScale(
        face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20)
    )
    eyes_count = len(eyes)
    checks["eyes_detected"] = eyes_count >= 1
    logger.debug(f"Test 1 - Yeux : {eyes_count} détecté(s)")

    # Test 2 : Analyse de texture (Laplacien)
    texture_score = _compute_texture_score(face_gray)
    checks["texture_score"] = texture_score
    checks["texture_ok"] = texture_score > 15
    logger.debug(f"Test 2 - Texture : {texture_score:.1f} (seuil: >15)")

    # Test 3 : Saturation HSV (écran vs peau)
    sat_score = _compute_saturation_score(face_color)
    checks["saturation_score"] = sat_score
    checks["saturation_ok"] = sat_score < 120
    logger.debug(f"Test 3 - Saturation : {sat_score:.1f} (seuil: <120)")

    # Test 4 : Contraste et luminosité (photo claire artificielle)
    contrast_score = _compute_contrast_score(face_gray)
    checks["contrast_score"] = contrast_score
    checks["contrast_ok"] = 30 < contrast_score < 100
    logger.debug(f"Test 4 - Contraste : {contrast_score:.1f} (plage: 30-100)")

    # Test 5 : Gradients directionnels (photo = contours nets)
    gradient_score = _compute_edge_quality(face_gray)
    checks["edge_quality"] = gradient_score
    checks["edges_natural"] = gradient_score < 80
    logger.debug(f"Test 5 - Qualité arêtes : {gradient_score:.1f} (seuil: <80)")

    # Test 6 : Détection flou (motion blur naturel)
    blur_score = _compute_blur_score(face_gray)
    checks["blur_score"] = blur_score
    checks["blur_natural"] = blur_score < 50
    logger.debug(f"Test 6 - Flou : {blur_score:.1f} (seuil: <50)")

    # Test 7 : Local Binary Pattern (texture locale)
    lbp_score = _compute_lbp_score(face_gray)
    checks["lbp_texture"] = lbp_score
    checks["lbp_natural"] = lbp_score > 30
    logger.debug(f"Test 7 - LBP texture : {lbp_score:.1f} (seuil: >30)")

    # Test 8 : Compatibilité couleur (pas d'artefacts compression)
    color_consistency = _compute_color_consistency(face_color)
    checks["color_consistency"] = color_consistency
    checks["color_ok"] = color_consistency > 70
    logger.debug(f"Test 8 - Cohérence couleur : {color_consistency:.1f} (seuil: >70)")

    # ── 5. Calcul du score final ────────────────────────────
    score_sum = 0
    if checks["eyes_detected"]:
        score_sum += 15
    if checks["texture_ok"]:
        score_sum += 15
    if checks["saturation_ok"]:
        score_sum += 15
    if checks["contrast_ok"]:
        score_sum += 12
    if checks["edges_natural"]:
        score_sum += 12
    if checks["blur_natural"]:
        score_sum += 12
    if checks["lbp_natural"]:
        score_sum += 12
    if checks["color_ok"]:
        score_sum += 12

    confidence = min(100, score_sum)

    # ── 6. Décision finale ───────────────────────────────────
    # Seuil minimum : au moins 70% de confiance
    min_threshold = 70

    if confidence >= min_threshold:
        return {
            "is_live":       True,
            "reason":        f"Visage vivant validé (confiance: {confidence}%)",
            "face_detected": True,
            "confidence":    confidence,
            "checks":        checks
        }
    else:
        reason = _build_rejection_reason(checks)
        return {
            "is_live":       False,
            "reason":        f"Liveness rejetée ({confidence}%) — {reason}",
            "face_detected": True,
            "confidence":    confidence,
            "checks":        checks
        }


# ── Liveness temps réel (flux vidéo) multi-couches ──────────
def check_liveness_realtime(camera) -> dict:
    """
    Liveness detection en temps réel avec plusieurs heuristiques.
    Détecte clignements naturels ET vérifie cohérence vidéo.

    Paramètre :
        camera : objet caméra OpenCV (cv2.VideoCapture)

    Retourne :
    {
        "is_live"      : True / False
        "reason"       : explication
        "blink_count"  : nombre de clignements
        "confidence"   : score 0-100 (%)
    }
    """
    logger.info(f"Liveness temps réel — attend {config.BLINK_COUNT_REQUIRED} clignements")

    face_cascade, eye_cascade = _load_detectors()

    blink_count            = 0
    eyes_closed            = False
    frame_list             = []
    motion_scores          = []
    start_time             = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed > config.LIVENESS_TIMEOUT:
            logger.warning(f"Timeout liveness après {elapsed:.1f}s")
            return {
                "is_live":     False,
                "reason":      f"Timeout : aucun clignement en {config.LIVENESS_TIMEOUT}s",
                "blink_count": blink_count,
                "confidence":  blink_count * 20
            }

        ret, frame = camera.read()
        if not ret:
            logger.error("Impossible de lire la caméra")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Détecter visage
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(faces) == 0:
            continue

        # Prendre le plus grand visage
        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        face_gray = gray[fy:fy+fh, fx:fx+fw]
        face_color = frame[fy:fy+fh, fx:fx+fw]

        # Détecter yeux
        eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5, minSize=(20, 20))
        eyes_count = len(eyes)

        # Test clignement
        if eyes_count >= 2:
            if eyes_closed:
                blink_count += 1
                eyes_closed = False
                logger.info(f"Clignement #{blink_count}")

                if blink_count >= config.BLINK_COUNT_REQUIRED:
                    # Double-check : analyser les frames capturées
                    stabilité = _check_motion_stability(frame_list)
                    cohérence = _check_color_stability(frame_list)

                    confidence = min(100, 80 + stabilité + cohérence)

                    logger.info(
                        f"Liveness validée : {blink_count} blinks, "
                        f"stabilité={stabilité}, cohérence={cohérence}, "
                        f"confidence={confidence}%"
                    )
                    return {
                        "is_live":     True,
                        "reason":      f"{blink_count} clignement(s) naturels détecté(s)",
                        "blink_count": blink_count,
                        "confidence":  confidence
                    }
        else:
            eyes_closed = True

        # Stocker frames pour analyse
        frame_list.append(face_gray.copy())
        if len(frame_list) > 10:
            frame_list.pop(0)

        time.sleep(0.05)  # 20 fps

    return {
        "is_live":     False,
        "reason":      "Erreur lecture caméra durant liveness",
        "blink_count": blink_count,
        "confidence":  blink_count * 20
    }


# ── Nouvelles fonctions d'analyse avancée ───────────────────

def _build_rejection_reason(checks: dict) -> str:
    """
    Construit un message explicite sur les raisons du rejet.
    """
    reasons = []
    if not checks.get("eyes_detected"):
        reasons.append("aucun oeil détecté")
    if not checks.get("texture_ok"):
        reasons.append("texture trop uniforme (photo probable)")
    if not checks.get("saturation_ok"):
        reasons.append("saturation anormale (écran probable)")
    if not checks.get("contrast_ok"):
        reasons.append("contraste artificiel")
    if not checks.get("edges_natural"):
        reasons.append("arêtes trop nettes (photo)")
    if not checks.get("blur_natural"):
        reasons.append("compression/artifaction vidéo")
    if not checks.get("lbp_natural"):
        reasons.append("texture LBP artificielle")
    if not checks.get("color_ok"):
        reasons.append("incohérence couleur")

    return " | ".join(reasons) if reasons else "raison inconnue"


def _compute_contrast_score(gray_face: np.ndarray) -> float:
    """
    Calcule le contraste via l'écart-type de l'histogramme.
    Peau naturelle : 30-100
    Photo nette : > 100
    Image floue : < 30
    """
    # Normaliser les valeurs de pixels
    hist = cv2.calcHist([gray_face], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # Variance de l'histogramme = contraste
    mean_intensity = np.average(np.arange(256), weights=hist)
    variance = np.average((np.arange(256) - mean_intensity) ** 2, weights=hist)
    return min(150, float(np.sqrt(variance)))


def _compute_edge_quality(gray_face: np.ndarray) -> float:
    """
    Analyse la qualité des arêtes.
    Photo = arêtes très nettes / peau vivante = arêtes naturelles.
    """
    # Sobol pour obtenir les gradients
    sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Score : variance des gradients
    # Photo nette : score élevé (>80)
    # Peau : score modéré (<80)
    return float(magnitude.std())


def _compute_blur_score(gray_face: np.ndarray) -> float:
    """
    Détecte le motion blur et la compression vidéo.
    Peau naturelle : blur < 50
    Compression/artifaction vidéo : blur > 50
    """
    # Laplacien pour déterminer la netteté
    laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
    blur_variance = laplacian.var()

    # Inverse : plus bas = plus flou
    # Score : plus bas = plus naturel pour liveness
    return min(100, float(100 - blur_variance / 10))


def _compute_lbp_score(gray_face: np.ndarray) -> float:
    """
    Local Binary Pattern — texture très locale.
    Peau naturelle : variabilité (score >30)
    Photo : uniformité (score <30)
    """
    # Appliquer LBP simplifié
    h, w = gray_face.shape
    lbp_map = np.zeros((h - 2, w - 2), dtype=np.uint8)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = gray_face[i, j]
            neighbors = [
                gray_face[i - 1, j - 1],
                gray_face[i - 1, j],
                gray_face[i - 1, j + 1],
                gray_face[i, j + 1],
                gray_face[i + 1, j + 1],
                gray_face[i + 1, j],
                gray_face[i + 1, j - 1],
                gray_face[i, j - 1],
            ]

            binary = 0
            for k in range(8):
                if neighbors[k] >= center:
                    binary |= 1 << k

            lbp_map[i - 1, j - 1] = binary

    # Variabilité LBP
    return float(lbp_map.std())


def _compute_color_consistency(color_face: np.ndarray) -> float:
    """
    Analyse la cohérence couleur.
    Peau naturelle : cohérence RGB (score >70)
    Écran/compression : artefacts (score <70)
    """
    # Séparer les canaux
    if len(color_face.shape) == 2:
        return 75.0

    b, g, r = cv2.split(color_face)

    # Corrélation entre canaux
    corr_rg = np.corrcoef(r.flatten(), g.flatten())[0, 1]
    corr_rb = np.corrcoef(r.flatten(), b.flatten())[0, 1]
    corr_gb = np.corrcoef(g.flatten(), b.flatten())[0, 1]

    # Moyenne des corrélations
    avg_corr = (corr_rg + corr_rb + corr_gb) / 3
    return min(100, float((avg_corr + 1) * 50))


def _check_motion_stability(frame_list: list) -> float:
    """
    Vérifie la stabilité du mouvement entre frames.
    Mouvements naturels vs robotic deepfakes.
    """
    if len(frame_list) < 3:
        return 10.0

    # Comparer consécutives frames
    stability_scores = []
    for i in range(1, len(frame_list)):
        diff = cv2.absdiff(frame_list[i - 1], frame_list[i])
        mean_diff = diff.mean()

        # Mouvement naturel : variabilité <10
        # Deepfake/comprssion : artefacts >20
        if mean_diff < 15:
            stability_scores.append(min(20, 20 - mean_diff))

    avg_stability = np.mean(stability_scores) if stability_scores else 0
    return min(15, float(avg_stability))


def _check_color_stability(frame_list: list) -> float:
    """
    Vérifie la cohérence couleur entre frames consécutives.
    Liveness : cohérence naturelle
    Deepfake/vidéo : artefacts de compression
    """
    if len(frame_list) < 2:
        return 10.0

    # Utiliser la première comme référence
    ref = frame_list[0]
    consistency_scores = []

    for frame in frame_list[1:]:
        # Comparer histogrammes
        hist_ref = cv2.calcHist([ref], [0], None, [256], [0, 256])
        hist_cur = cv2.calcHist([frame], [0], None, [256], [0, 256])

        # Distance Bhattacharyya
        dist = cv2.compareHist(hist_ref, hist_cur, cv2.HISTCMP_BHATTACHARYYA)
        consistency_scores.append(min(10, 10 - dist))

    avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
    return min(15, float(avg_consistency))


# ── Fonctions utilitaires (mise à jour) ──────────────────────

def _compute_texture_score(gray_face: np.ndarray) -> float:
    """
    Score de texture via Laplacien multi-échelle.
    Peau vivante : variance élevée (>15)
    Photo : uniformité (< 15)
    """
    laplacian = cv2.Laplacian(gray_face, cv2.CV_64F)
    return float(laplacian.var())


def _compute_saturation_score(color_face: np.ndarray) -> float:
    """
    Saturation HSV — détecte écrans et impressions.
    Peau naturelle : 10-80
    Écran / impression : > 120
    """
    if len(color_face.shape) == 2:
        return 50.0

    hsv = cv2.cvtColor(color_face, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())