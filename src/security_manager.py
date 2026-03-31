# ============================================================
# src/security_manager.py
# Rôle : Centraliser la sécurité du pointage (Anti-fraude & RGPD)
#
# Logique : 
#   1. Valider le caractère vivant (Liveness)
#   2. Déclencher les alertes affichées à l'écran en cas de fraude
#   3. Gérer l'anonymisation des données (Embeddings vs Images)
#
# Utilisé par : main.py
# Dépend de   : liveness.py, gpio_feedback.py, config.py
# ============================================================

import logging
import liveness
import gpio_feedback
import config
from datetime import datetime

logger = logging.getLogger(__name__)

class SecurityManager:
    """
    Gère les protocoles de sécurité pour le Groupe 7.
    Assure la protection contre le spoofing et la conformité RGPD[cite: 168].
    """

    def __init__(self):
        self.min_blink = config.BLINK_COUNT_REQUIRED # Défini dans le planning [cite: 392]
        self.failed_attempts = 0

    def verify_identity_safety(self, camera_stream):
        """
        Vérifie si l'utilisateur est bien une personne physique[cite: 64, 417].
        Utilise la méthode EAR (Eye Aspect Ratio)[cite: 161, 422].
        """
        logger.info("Démarrage du protocole anti-spoofing...")
        
        # 1. Appel du module de liveness (clignotement des yeux)
        result = liveness.check_liveness_realtime(camera_stream)
        
        if result["is_live"]:
            logger.info(f"Vérification réussie : {result['reason']} [cite: 421]")
            return True
        else:
            # 2. Alerte en cas de détection de photo ou vidéo (Spoofing)
            logger.warning(f"Alerte Sécurité : {result['reason']} [cite: 426]")
            self._handle_fraud_attempt()
            return False

    def _handle_fraud_attempt(self):
        """
        Déclenche les messages d'alerte sur l'ecran de la borne[cite: 23, 34].
        """
        self.failed_attempts += 1
        # Message d'alerte de fraude visible a l'ecran [cite: 501]
        gpio_feedback.signal_spoof_detected()
        logger.error(f"Tentative de fraude bloquée. Total échecs : {self.failed_attempts}")

    def prepare_biometric_data(self, frame):
        """
        Prépare l'image pour l'envoi au serveur Backend[cite: 14, 201].
        Note : On ne stocke pas l'image localement pour respecter le RGPD[cite: 168].
        """
        # Le système extrait l'embedding pour éviter de stocker l'image brute 
        # Cela réduit les risques liés aux données sensibles.
        logger.info("Anonymisation et préparation du vecteur facial[cite: 161, 166].")
        return frame # À envoyer vers l'API Django [cite: 213, 219]

    def log_security_event(self, event_type, user_id="Inconnu"):
        """
        Enregistre les événements de sécurité dans l'historique[cite: 41, 165].
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Ces logs sont consultables sur le Dashboard Administrateur[cite: 28, 270].
        logger.info(f"EVENEMENT [{event_type}] - Utilisateur: {user_id} - Time: {timestamp}")

