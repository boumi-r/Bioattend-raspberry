# ============================================================
# src/pir.py
# Rôle : détecter la présence d'une personne via le capteur
#        PIR (Passive Infrared) connecté sur GPIO 17
#
# Fonctionnement du PIR HC-SR501 :
#   - GPIO HIGH (1) → mouvement détecté
#   - GPIO LOW  (0) → pas de mouvement
#   - Délai de chauffe : 30-60 secondes au démarrage
#   - Portée : 3 à 7 mètres
#   - Angle : 120 degrés
#
# Utilisé par : main.py
# Dépend de   : config.py, gpio_feedback.py
# ============================================================
import time
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import config

logger = logging.getLogger(__name__)


# ── Chargement GPIO (réel ou simulé) ─────────────────────────
try:
    import RPi.GPIO as GPIO
    IS_RASPBERRY = True
    logger.info("RPi.GPIO chargé — mode Raspberry Pi réel")

except (ImportError, RuntimeError):
    IS_RASPBERRY = False
    logger.warning("RPi.GPIO non disponible — mode simulation PIR activé")

    class GPIO:
        """Mock RPi.GPIO pour Codespaces"""
        BCM  = "BCM"
        IN   = "IN"
        OUT  = "OUT"
        HIGH = True
        LOW  = False
        PUD_DOWN = "PUD_DOWN"

        @staticmethod
        def setmode(mode):
            logger.debug(f"[MOCK GPIO] setmode({mode})")

        @staticmethod
        def setwarnings(flag):
            pass

        @staticmethod
        def setup(pin, mode, **kwargs):
            logger.debug(f"[MOCK GPIO] setup(pin={pin}, mode={mode})")

        @staticmethod
        def input(pin):
            # Simule toujours "pas de mouvement" par défaut
            logger.debug(f"[MOCK GPIO] input(pin={pin}) → LOW (simulation)")
            return False

        @staticmethod
        def cleanup():
            logger.debug("[MOCK GPIO] cleanup()")


# ── Setup du capteur PIR ──────────────────────────────────────
def setup():
    """
    Configure le pin GPIO du capteur PIR en mode entrée.
    Doit être appelée au démarrage avant wait_for_motion().

    PUD_DOWN = résistance pull-down interne activée
    → évite les lectures parasites quand le capteur est LOW
    """
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(
        config.GPIO_PIR,
        GPIO.IN,
        pull_up_down=GPIO.PUD_DOWN
    )
    logger.info(f"Capteur PIR configuré sur GPIO {config.GPIO_PIR}")


# ── Attente d'une détection ───────────────────────────────────
def wait_for_motion(timeout: int = None) -> bool:
    """
    Attend qu'une présence soit détectée par le PIR.
    Bloque jusqu'à détection ou timeout.

    Paramètre :
        timeout : secondes max d'attente (None = infini)

    Retourne :
        True  → mouvement détecté
        False → timeout écoulé sans détection
    """
    logger.info(f"En attente de présence (GPIO {config.GPIO_PIR})...")

    start_time = time.time()

    while True:
        # Lire l'état du capteur PIR
        pir_state = GPIO.input(config.GPIO_PIR)

        if pir_state:
            # GPIO HIGH → présence détectée !
            logger.info("Présence détectée par le capteur PIR !")
            return True

        # Vérifier le timeout si défini
        if timeout is not None:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                logger.info(f"Timeout PIR après {elapsed:.1f}s — aucune présence")
                return False

        # Petite pause pour ne pas saturer le CPU
        time.sleep(0.1)   # vérification 10 fois par seconde


# ── Lecture instantanée ───────────────────────────────────────
def is_motion_detected() -> bool:
    """
    Lit l'état actuel du capteur PIR sans bloquer.

    Retourne :
        True  → mouvement détecté en ce moment
        False → pas de mouvement
    """
    state = GPIO.input(config.GPIO_PIR)
    if state:
        logger.debug("PIR : mouvement détecté")
    return bool(state)


# ── Attente que le PIR se stabilise ──────────────────────────
def wait_stable(stable_duration: float = 1.0) -> bool:
    """
    Attend que le PIR soit stable (pas de faux positif).
    Vérifie que le signal reste HIGH pendant stable_duration.

    Utile pour éviter les déclenchements parasites.

    Paramètre :
        stable_duration : secondes de stabilité requises

    Retourne :
        True  → signal stable → vraie détection
        False → signal instable → faux positif
    """
    logger.debug(f"Vérification stabilité PIR ({stable_duration}s)...")

    start = time.time()
    while time.time() - start < stable_duration:
        if not GPIO.input(config.GPIO_PIR):
            # Signal est retombé → faux positif
            logger.debug("PIR instable — faux positif ignoré")
            return False
        time.sleep(0.05)

    logger.info("PIR stable — présence confirmée")
    return True


# ── Simulation pour tests ─────────────────────────────────────
def simulate_detection():
    """
    Simule une détection PIR pour les tests sur Codespaces.
    Ne fonctionne que si IS_RASPBERRY = False.
    """
    if IS_RASPBERRY:
        logger.warning("simulate_detection() ignoré sur vrai Pi")
        return

    logger.info("[SIMULATION] Présence simulée par le PIR")