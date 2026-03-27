# ============================================================
# src/gpio_feedback.py
# Rôle : contrôler les LEDs et le buzzer via GPIO
#
# Sur Raspberry Pi  → utilise RPi.GPIO (vrais composants)
# Sur Codespaces    → utilise un mock (simulation dans logs)
#
# Utilisé par : main.py, pir.py, liveness.py
# Dépend de   : config.py
# ============================================================
import time
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import config

logger = logging.getLogger(__name__)


# ── Chargement de GPIO (réel ou simulé) ─────────────────────
# On essaie d'importer RPi.GPIO
# Si on n'est pas sur un Pi → on charge le mock à la place
try:
    import RPi.GPIO as GPIO
    IS_RASPBERRY = True
    logger.info("RPi.GPIO chargé — mode Raspberry Pi réel")

except (ImportError, RuntimeError):
    # On n'est pas sur un Pi — on crée un mock simple
    IS_RASPBERRY = False
    logger.warning("RPi.GPIO non disponible — mode simulation activé")

    class GPIO:
        """
        Mock de RPi.GPIO pour développement sur PC/Codespaces.
        Simule toutes les fonctions sans matériel réel.
        """
        BCM  = "BCM"
        OUT  = "OUT"
        HIGH = True
        LOW  = False

        @staticmethod
        def setmode(mode):
            logger.debug(f"[MOCK GPIO] setmode({mode})")

        @staticmethod
        def setwarnings(flag):
            logger.debug(f"[MOCK GPIO] setwarnings({flag})")

        @staticmethod
        def setup(pin, mode):
            logger.debug(f"[MOCK GPIO] setup(pin={pin}, mode={mode})")

        @staticmethod
        def output(pin, state):
            state_str = "HIGH" if state else "LOW"
            logger.debug(f"[MOCK GPIO] output(pin={pin}, state={state_str})")

        @staticmethod
        def cleanup():
            logger.debug("[MOCK GPIO] cleanup()")


# ── Initialisation GPIO ──────────────────────────────────────
def setup():
    """
    Configure les pins GPIO en mode sortie.
    Doit être appelée UNE SEULE FOIS au démarrage du script.

    Pins configurés :
        GPIO_LED_GREEN → sortie (LED verte)
        GPIO_LED_RED   → sortie (LED rouge)
        GPIO_BUZZER    → sortie (Buzzer)
    """
    GPIO.setmode(GPIO.BCM)       # numérotation BCM (pas BOARD)
    GPIO.setwarnings(False)      # désactiver les warnings GPIO

    GPIO.setup(config.GPIO_LED_GREEN, GPIO.OUT)
    GPIO.setup(config.GPIO_LED_RED,   GPIO.OUT)
    GPIO.setup(config.GPIO_BUZZER,    GPIO.OUT)

    # S'assurer que tout est éteint au démarrage
    GPIO.output(config.GPIO_LED_GREEN, GPIO.LOW)
    GPIO.output(config.GPIO_LED_RED,   GPIO.LOW)
    GPIO.output(config.GPIO_BUZZER,    GPIO.LOW)

    logger.info("GPIO initialisé — LEDs et Buzzer prêts")


# ── Nettoyage GPIO ───────────────────────────────────────────
def cleanup():
    """
    Remet tous les pins à LOW et libère le GPIO.
    Doit être appelée à la fin du script (dans finally).
    """
    GPIO.output(config.GPIO_LED_GREEN, GPIO.LOW)
    GPIO.output(config.GPIO_LED_RED,   GPIO.LOW)
    GPIO.output(config.GPIO_BUZZER,    GPIO.LOW)
    GPIO.cleanup()
    logger.info("GPIO nettoyé")


# ── Signaux visuels et sonores ───────────────────────────────
def signal_access_granted():
    """
    Accès autorisé :
    - LED verte allumée pendant LED_DURATION secondes
    - Buzzer long (1 bip long)
    """
    logger.info("ACCÈS AUTORISÉ → LED verte + buzzer long")

    # Allumer LED verte + buzzer
    GPIO.output(config.GPIO_LED_GREEN, GPIO.HIGH)
    GPIO.output(config.GPIO_BUZZER,    GPIO.HIGH)
    time.sleep(config.BUZZER_LONG)

    # Éteindre buzzer mais garder LED verte
    GPIO.output(config.GPIO_BUZZER, GPIO.LOW)
    time.sleep(config.LED_DURATION - config.BUZZER_LONG)

    # Éteindre LED verte
    GPIO.output(config.GPIO_LED_GREEN, GPIO.LOW)


def signal_access_denied():
    """
    Accès refusé :
    - LED rouge allumée pendant LED_DURATION secondes
    - Buzzer court (2 bips courts)
    """
    logger.info("ACCÈS REFUSÉ → LED rouge + 2 bips courts")

    GPIO.output(config.GPIO_LED_RED, GPIO.HIGH)

    # 2 bips courts
    for _ in range(2):
        GPIO.output(config.GPIO_BUZZER, GPIO.HIGH)
        time.sleep(config.BUZZER_SHORT)
        GPIO.output(config.GPIO_BUZZER, GPIO.LOW)
        time.sleep(config.BUZZER_SHORT)

    # Garder LED rouge le reste du temps
    time.sleep(config.LED_DURATION - (4 * config.BUZZER_SHORT))
    GPIO.output(config.GPIO_LED_RED, GPIO.LOW)


def signal_processing():
    """
    Traitement en cours :
    - LED rouge clignotante lente (3 fois)
    - Indique que le Pi travaille
    """
    logger.info("Traitement en cours → LED rouge clignotante")

    for _ in range(3):
        GPIO.output(config.GPIO_LED_RED, GPIO.HIGH)
        time.sleep(0.3)
        GPIO.output(config.GPIO_LED_RED, GPIO.LOW)
        time.sleep(0.3)


def signal_spoof_detected():
    """
    Attaque spoof détectée (photo ou vidéo présentée) :
    - LED rouge clignotante rapide (5 fois)
    - 3 bips courts rapides
    """
    logger.warning("SPOOF DÉTECTÉ → LED rouge rapide + 3 bips")

    for _ in range(5):
        GPIO.output(config.GPIO_LED_RED, GPIO.HIGH)
        GPIO.output(config.GPIO_BUZZER,  GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(config.GPIO_LED_RED, GPIO.LOW)
        GPIO.output(config.GPIO_BUZZER,  GPIO.LOW)
        time.sleep(0.1)


def signal_error():
    """
    Erreur système (serveur inaccessible, etc.) :
    - LED rouge allumée fixe 2 secondes
    - 1 bip long
    """
    logger.error("ERREUR SYSTÈME → LED rouge fixe")

    GPIO.output(config.GPIO_LED_RED, GPIO.HIGH)
    GPIO.output(config.GPIO_BUZZER,  GPIO.HIGH)
    time.sleep(1.0)
    GPIO.output(config.GPIO_BUZZER,  GPIO.LOW)
    time.sleep(1.0)
    GPIO.output(config.GPIO_LED_RED, GPIO.LOW)


def signal_ready():
    """
    Système prêt et en attente :
    - LED verte clignote 2 fois rapidement
    - Indique que le Pi est prêt à détecter
    """
    logger.info("Système prêt → LED verte 2 clignements")

    for _ in range(2):
        GPIO.output(config.GPIO_LED_GREEN, GPIO.HIGH)
        time.sleep(0.2)
        GPIO.output(config.GPIO_LED_GREEN, GPIO.LOW)
        time.sleep(0.2)