# ============================================================
# src/api_client.py
# Rôle : envoyer l'embedding au serveur Django et récupérer
#        la décision d'identification
#
# Utilisé par : main.py
# Dépend de   : config.py
# ============================================================
import json
import requests
import logging
import sys
import os

# Ajouter le dossier src au path pour importer config
sys.path.insert(0, os.path.dirname(__file__))
import config

# Logger pour afficher les messages dans le terminal
logger = logging.getLogger(__name__)


def check_server() -> bool:
    """Vérifie que le serveur Django est accessible."""
    try:
        response = requests.get(config.SERVER_URL, timeout=10)
        return response.status_code < 500
    except requests.exceptions.RequestException:
        return False


def send_embedding(embedding: list, bbox: list) -> dict:
    """
    Envoie l'embedding au serveur Django via HTTP POST.

    Paramètres :
        embedding : liste de 512 floats (vecteur InsightFace)
        bbox      : [x1, y1, x2, y2] du visage détecté

    Retourne un dictionnaire avec la réponse du serveur.
    """

    headers = {"Content-Type": "application/json"}
    if config.API_TOKEN:
        headers["Authorization"] = f"Bearer {config.API_TOKEN}"

    payload = {
        "embedding": embedding,
        "bbox": bbox,
    }

    try:
        logger.info(f"Envoi embedding vers {config.API_ENDPOINT}...")

        response = requests.post(
            url     = config.API_ENDPOINT,
            json    = payload,
            headers = headers,
            timeout = 60,
        )

        logger.info(f"Réponse reçue — HTTP {response.status_code}")

        data = response.json()

        if config.DEBUG:
            logger.debug(f"Réponse serveur : {data}")

        return data

    except requests.exceptions.ConnectionError:
        logger.error(f"Impossible de joindre le serveur : {config.API_ENDPOINT}")
        return {
            "success": False,
            "error":   f"Serveur inaccessible : {config.API_ENDPOINT}"
        }

    except requests.exceptions.Timeout:
        logger.error("Le serveur n'a pas répondu à temps.")
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
        logger.error(f"Réponse invalide du serveur (pas du JSON) : {e}")
        return {
            "success": False,
            "error":   "Réponse serveur invalide."
        }

