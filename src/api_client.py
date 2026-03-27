# ============================================================
# src/api_client.py
# Rôle : envoyer l'image au serveur Django et récupérer
#        les embeddings + la décision d'accès
#
# Utilisé par : main.py
# Dépend de   : config.py
# ============================================================
import requests
import logging
import sys
import os

# Ajouter le dossier src au path pour importer config
sys.path.insert(0, os.path.dirname(__file__))
import config

# Logger pour afficher les messages dans le terminal
logger = logging.getLogger(__name__)


def send_image(image_bytes: bytes) -> dict:
    """
    Envoie une image au serveur Django via HTTP POST.

    Paramètre :
        image_bytes : les bytes de l'image capturée par la caméra

    Retourne un dictionnaire :
        {
            "success"      : True / False
            "face_detected": True / False
            "embedding"    : liste de 512 floats (si succès)
            "bbox"         : [x1, y1, x2, y2] (si succès)
            "error"        : message d'erreur (si échec)
        }
    """

    # ── 1. Préparer les headers HTTP ────────────────────────────
    # Le token permet au serveur Django de vérifier
    # que c'est bien le Pi qui envoie la requête
    headers = {}
    if config.API_TOKEN:
        headers["Authorization"] = f"Token {config.API_TOKEN}"

    # ── 2. Préparer le fichier image ────────────────────────────
    # On envoie l'image comme un fichier multipart/form-data
    # C'est exactement ce que Django attend dans request.FILES
    files = {
        "image": ("capture.jpg", image_bytes, "image/jpeg")
        #           ↑ nom       ↑ bytes       ↑ type MIME
    }

    # ── 3. Envoyer la requête POST ──────────────────────────────
    try:
        logger.info(f"Envoi image vers {config.API_ENDPOINT}...")

        response = requests.post(
            url     = config.API_ENDPOINT,
            files   = files,
            headers = headers,
            timeout = 60   # secondes — évite de bloquer indéfiniment
        )

        # ── 4. Analyser la réponse ───────────────────────────────
        logger.info(f"Réponse reçue — HTTP {response.status_code}")

        # Convertir la réponse JSON en dictionnaire Python
        data = response.json()

        if config.DEBUG:
            if data.get("success"):
                logger.debug(f"Embedding reçu : {len(data.get('embedding', []))} valeurs")
                logger.debug(f"Bbox : {data.get('bbox')}")
            else:
                logger.debug(f"Erreur serveur : {data.get('error')}")

        return data

    # ── 5. Gestion des erreurs réseau ────────────────────────────
    except requests.exceptions.ConnectionError:
        logger.error(f"Impossible de joindre le serveur : {config.API_ENDPOINT}")
        logger.error("Vérifie que le serveur Django est démarré et que l'IP est correcte.")
        return {
            "success": False,
            "error":   f"Serveur inaccessible : {config.API_ENDPOINT}"
        }

    except requests.exceptions.Timeout:
        logger.error("Le serveur n'a pas répondu dans les 10 secondes.")
        return {
            "success": False,
            "error":   "Timeout — le serveur met trop de temps à répondre."
        }

    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur réseau inattendue : {e}")
        return {
            "success": False,
            "error":   f"Erreur réseau : {str(e)}"
        }

    except ValueError as e:
        # ValueError si la réponse n'est pas du JSON valide
        logger.error(f"Réponse invalide du serveur (pas du JSON) : {e}")
        return {
            "success": False,
            "error":   "Réponse serveur invalide."
        }


def check_server() -> bool:
    """
    Vérifie que le serveur Django est accessible.
    Appelée au démarrage du script principal.

    Retourne True si le serveur répond, False sinon.
    """
    try:
        # On fait un GET simple sur la racine du serveur
        response = requests.get(
            config.SERVER_URL,
            timeout=5
        )
        logger.info(f"Serveur accessible — HTTP {response.status_code}")
        return True

    except requests.exceptions.ConnectionError:
        logger.error(f"Serveur inaccessible : {config.SERVER_URL}")
        return False

    except requests.exceptions.Timeout:
        logger.error("Serveur trop lent à répondre.")
        return False