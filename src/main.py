import cv2
import logging
import gpio_feedback
from security_manager import SecurityManager
import requests # Pour l'API Django/Supabase
import config

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Initialisation des composants matériels
    gpio_feedback.setup() [cite: 308]
    security = SecurityManager()
    camera = cv2.VideoCapture(0) # Utilisation d'OpenCV [cite: 138, 155]
    
    logger.info("Système BioAttend prêt et en attente de mouvement...")
    gpio_feedback.signal_ready()

    try:
        while True:
            # simuler l'attente du capteur PIR [cite: 64, 65]
            # if pir_detected(): 
            
            logger.info("Mouvement détecté ! Activation de la phase de sécurité.")
            
            # 2. Phase de Sécurité : Liveness Detection (Clignotement)
            # On vérifie si c'est un humain avant tout calcul lourd [cite: 20, 158]
            if security.verify_identity_safety(camera):
                
                # 3. Capture et Envoi au Backend
                gpio_feedback.signal_processing()
                ret, frame = camera.read()
                
                if ret:
                    # Préparation des données selon le RGPD (Vecteurs/Embeddings) [cite: 161, 168]
                    processed_frame = security.prepare_biometric_data(frame)
                    
                    # 4. Identification via API REST (Django -> InsightFace -> Supabase)
                    # On envoie l'image au serveur distant pour le matchmaking [cite: 200, 204]
                    try:
                        response = requests.post(config.API_URL, files={'image': processed_frame})
                        result = response.json() [cite: 207]
                        
                        if result.get("authenticated"):
                            # 5. Succès : Pointage enregistré [cite: 22, 165]
                            logger.info(f"Bienvenue {result['user_name']}")
                            gpio_feedback.signal_access_granted()
                        else:
                            # Échec : Utilisateur inconnu dans la DB [cite: 428]
                            logger.warning("Utilisateur non reconnu.")
                            gpio_feedback.signal_access_denied()
                            
                    except Exception as e:
                        logger.error(f"Erreur de connexion au serveur : {e}")
                        gpio_feedback.signal_error()
            else:
                # Fraude détectée (Photo/Vidéo)
                logger.warning("Accès bloqué : Tentative de spoofing suspectée.")
                # Le signal d'alerte est déjà géré dans SecurityManager

    except KeyboardInterrupt:
        logger.info("Arrêt du système par l'utilisateur.")
    finally:
        # Nettoyage pour éviter les surchauffes ou courts-circuits [cite: 114]
        camera.release()
        gpio_feedback.cleanup()

if __name__ == "__main__":
    main()