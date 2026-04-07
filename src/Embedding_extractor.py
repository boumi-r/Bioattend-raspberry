# ============================================================
# src/embedding_extractor.py
# Rôle : extraire un embedding facial avec InsightFace
#        directement sur le Raspberry Pi
#
# Utilisé par : main.py (avant l'envoi API)
# Install     : pip install insightface onnxruntime
# ============================================================
 
import logging
import numpy as np
import cv2
 
logger = logging.getLogger(__name__)
 
# InsightFace est importé à la demande pour ne pas crasher
# si le paquet n'est pas installé
try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.error("InsightFace non installé — pip install insightface onnxruntime")
 
 
class EmbeddingExtractor:
    """
    Extrait un vecteur d'embedding facial (512 floats)
    à partir d'une image OpenCV, via InsightFace (ArcFace).
 
    Usage :
        extractor = EmbeddingExtractor()
        embedding = extractor.extract(img)  # numpy array (512,) ou None
    """
 
    def __init__(self, model_name: str = "buffalo_sc"):
        """
        Initialise InsightFace.
 
        buffalo_sc  → modèle léger, recommandé pour Raspberry Pi
        buffalo_l   → modèle complet, plus précis mais plus lent
        """
        if not INSIGHTFACE_AVAILABLE:
            raise RuntimeError(
                "InsightFace non installé.\n"
                "Installe-le avec : pip install insightface onnxruntime"
            )
 
        logger.info(f"Chargement du modèle InsightFace : {model_name}...")
 
        # ctx_id=-1 → forcer CPU (pas de GPU sur Pi)
        self.app = FaceAnalysis(
            name=model_name,
            providers=["CPUExecutionProvider"]
        )
        # det_size : taille de détection — (320, 320) est plus rapide sur Pi
        self.app.prepare(ctx_id=-1, det_size=(320, 320))
 
        logger.info("InsightFace prêt.")
 
    def extract(self, img: np.ndarray) -> np.ndarray | None:
        """
        Détecte le visage principal et retourne son embedding.
 
        Paramètre :
            img : image OpenCV (BGR, uint8)
 
        Retourne :
            numpy array de shape (512,) si un visage est trouvé
            None si aucun visage détecté
        """
        if img is None or img.size == 0:
            logger.warning("Image vide passée à EmbeddingExtractor.extract()")
            return None
 
        # InsightFace attend du BGR — c'est déjà le format OpenCV
        faces = self.app.get(img)
 
        if not faces:
            logger.warning("Aucun visage détecté par InsightFace")
            return None
 
        if len(faces) > 1:
            logger.info(f"{len(faces)} visages détectés — sélection du plus grand")
 
        # Sélectionner le visage avec la plus grande bounding box
        main_face = max(faces, key=lambda f: _bbox_area(f.bbox))
 
        embedding = main_face.embedding  # numpy array (512,)
        logger.info(f"Embedding extrait — {len(embedding)} valeurs")
        logger.debug(f"Score de détection : {main_face.det_score:.4f}")
 
        return embedding
 
    def extract_to_list(self, img: np.ndarray) -> list[float] | None:
        """
        Comme extract(), mais retourne une liste Python (pour JSON).
        C'est ce format qui doit être envoyé au serveur Django.
        """
        embedding = self.extract(img)
        if embedding is None:
            return None
        return embedding.tolist()
 
 
def _bbox_area(bbox) -> float:
    """Calcule l'aire d'une bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)