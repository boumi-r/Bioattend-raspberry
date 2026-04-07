# ============================================================
# src/api_client.py
# Rôle : envoyer l'embedding au serveur Django et récupérer
#        la décision d'accès
#
# Utilisé par : main.py
# Dépend de   : config.py
# ============================================================
import requests
import logging
import sys
import os
from typing import Optional
 
sys.path.insert(0, os.path.dirname(__file__))
import config
 
logger = logging.getLogger(__name__)
 
 
def send_embedding(embedding: list[float]) -> dict:
    """
    Envoie un vecteur d'embedding au serveur Django via HTTP POST.
 
    Paramètre :
        embedding : liste de 512 floats extraits par InsightFace
 
    Retourne un dictionnaire :
        {
            "success"  : True / False
            "user"     : { "id": ..., "name": ... }  (si reconnu)
            "distance" : float  (si reconnu)
            "error"    : str    (si échec)
        }
    """
 
    # ── 1. Préparer les headers HTTP ────────────────────────────
    headers = {"Content-Type": "application/json"}
    if config.API_TOKEN:
        headers["Authorization"] = f"Bearer {config.API_TOKEN}"
 
    # ── 2. Préparer le payload JSON ─────────────────────────────
    payload = {
        "embedding": embedding   # liste de 512 floats
    }
 
    # ── 3. Envoyer la requête POST ──────────────────────────────
    try:
        logger.info(f"Envoi embedding vers {config.API_ENDPOINT}...")
 
        response = requests.post(
            url     = config.API_ENDPOINT,
            json    = payload,          # sérialise automatiquement en JSON
            headers = headers,
            timeout = 10
        )
 
        # ── 4. Analyser la réponse ───────────────────────────────
        logger.info(f"Réponse reçue — HTTP {response.status_code}")
        data = response.json()
 
        if config.DEBUG:
            if data.get("success"):
                logger.debug(f"Utilisateur reconnu : {data.get('user')}")
                logger.debug(f"Distance           : {data.get('distance')}")
            else:
                logger.debug(f"Erreur serveur : {data.get('error')}")
 
        return data
 
    # ── 5. Gestion des erreurs réseau ────────────────────────────
    except requests.exceptions.ConnectionError:
        logger.error(f"Impossible de joindre le serveur : {config.API_ENDPOINT}")
        return {"success": False, "error": f"Serveur inaccessible : {config.API_ENDPOINT}"}
 
    except requests.exceptions.Timeout:
        logger.error("Le serveur n'a pas répondu dans les 10 secondes.")
        return {"success": False, "error": "Timeout — serveur trop lent."}
 
    except requests.exceptions.RequestException as e:
        logger.error(f"Erreur réseau inattendue : {e}")
        return {"success": False, "error": f"Erreur réseau : {str(e)}"}
 
    except ValueError as e:
        logger.error(f"Réponse invalide du serveur (pas du JSON) : {e}")
        return {"success": False, "error": "Réponse serveur invalide."}
 
 
def check_server() -> bool:
    """
    Vérifie que le serveur Django est accessible.
    Appelée au démarrage du script principal.
    """
    try:
        response = requests.get(config.SERVER_URL, timeout=5)
        logger.info(f"Serveur accessible — HTTP {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        logger.error(f"Serveur inaccessible : {config.SERVER_URL}")
        return False
    except requests.exceptions.Timeout:
        logger.error("Serveur trop lent à répondre.")
        return False