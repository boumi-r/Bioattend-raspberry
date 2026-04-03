import cv2
import numpy as np
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

inside = None


def init_model():

    global inside
    inside = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    inside.prepare(ctx_id=-1, det_size=(640, 640))
    logger.info("Modèle InsightFace chargé")


def get_embedding(image_bytes: bytes) -> dict:
   

    if inside is None:
        raise RuntimeError("Modèle non chargé, init_model()")

    image_recu = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_recu, cv2.IMREAD_COLOR)

    if img is None:
        return {"success": False, "error": "Image invalide "}

    faces = inside.get(img)

    if len(faces) == 0:
        return {"success": False, "error": "Aucun visage détecté par InsightFace"}

    
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    embedding = face.embedding.tolist()
    bbox = face.bbox.astype(int).tolist()

    logger.info("Embedding extrait : %d valeurs, bbox=%s", len(embedding), bbox)
    return {
        "success": True,
        "embedding": embedding,
        "bbox": bbox, #[x1, y1, x2, y2]
    }
