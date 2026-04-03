
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import config

logger = logging.getLogger(__name__)


def _display_status(title: str, message: str, *, level: str = "info", hold_seconds: float = 0.0):
    
    border = "=" * 55
    screen_message = f"\n{border}\n[{title}]\n{message}\n{border}"

    print(screen_message, flush=True)

    log_method = getattr(logger, level, logger.info)
    log_method(f"{title} - {message}")

    if hold_seconds > 0:
        time.sleep(hold_seconds)


def setup():
   
    _display_status("INITIALISATION","ecran actif")


def cleanup():
    _display_status("ARRET", "Arret du systeme BioAttend.")


def signal_access_granted():
    _display_status(
        "ACCES AUTORISE", "Visage reconnu. Bienvenue !",
        hold_seconds=max(config.FEEDBACK_DURATION, 0),
    )


def signal_access_denied():
    _display_status(
        "ACCES REFUSE", "Visage non reconnu. Acces refuse.",
        level="warning",
        hold_seconds=max(config.FEEDBACK_DURATION, 0),
    )


def signal_processing():

    _display_status(
        "TRAITEMENT EN COURS",
        "Presence detectee. Analyse du visage et verification de vivacite en cours...",
        hold_seconds=0.6,
    )


def signal_spoof_detected():
    
    _display_status(
        "ALERTE SECURITE",
        "Tentative de fraude detectee : Verification bloquee.",
        level="warning",
        hold_seconds=1.0,
    )


def signal_error():
    
    _display_status(
        "ERREUR SYSTEME",
        "Une erreur est survenue. Reessayez.",
        level="error",
        hold_seconds=2.0,
    )


def signal_ready():
    
    _display_status(
        "SYSTEME PRET",
        "Placez-vous devant la camera pour demarrer la verification.",
        hold_seconds=0.4,
    )