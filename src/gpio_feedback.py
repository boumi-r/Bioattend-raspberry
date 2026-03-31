# ============================================================
# src/gpio_feedback.py
# Rôle : afficher des messages de statut sur l'ecran.
#
# Le projet utilisait initialement des LEDs et un buzzer via GPIO.
# Sur le materiel actuellement deploye, ces composants ne sont pas
# presents : on conserve donc la meme API publique mais on remplace
# les signaux physiques par des messages descriptifs visibles a l'ecran.
#
# Utilise par : main.py, pir.py, liveness.py
# Depent de   : config.py
# ============================================================
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
import config

logger = logging.getLogger(__name__)


def _display_status(title: str, message: str, *, level: str = "info", hold_seconds: float = 0.0):
    """
    Affiche un etat lisible sur l'ecran et dans les logs.

    Le rendu passe par stdout afin d'etre visible sur l'ecran branche
    au Raspberry, tout en restant exploitable dans les journaux.
    """
    border = "=" * 55
    screen_message = f"\n{border}\n[{title}]\n{message}\n{border}"

    print(screen_message, flush=True)

    log_method = getattr(logger, level, logger.info)
    log_method(f"{title} - {message}")

    if hold_seconds > 0:
        time.sleep(hold_seconds)


def setup():
    """
    Initialise le mode d'affichage des statuts a l'ecran.
    """
    _display_status(
        "INITIALISATION",
        "Mode ecran actif : les retours LED et buzzer sont remplaces par des messages descriptifs.",
    )


def cleanup():
    """
    Termine proprement l'affichage des statuts.
    """
    _display_status("ARRET", "Arret du systeme BioAttend.")


def signal_access_granted():
    """
    Affiche un message de succes apres validation de l'acces.
    """
    _display_status(
        "ACCES AUTORISE",
        "Identite validee. La personne est reconnue et l'acces est autorise.",
        hold_seconds=max(config.LED_DURATION, 0),
    )


def signal_access_denied():
    """
    Affiche un message de refus d'acces.
    """
    _display_status(
        "ACCES REFUSE",
        "Authentification impossible ou visage non reconnu. Acces refuse.",
        level="warning",
        hold_seconds=max(config.LED_DURATION, 0),
    )


def signal_processing():
    """
    Affiche que le systeme analyse la presence detectee.
    """
    _display_status(
        "TRAITEMENT EN COURS",
        "Presence detectee. Analyse du visage et verification de vivacite en cours...",
        hold_seconds=0.6,
    )


def signal_spoof_detected():
    """
    Affiche une alerte claire lorsqu'une tentative de fraude est detectee.
    """
    _display_status(
        "ALERTE SECURITE",
        "Tentative de fraude detectee : photo, video ou ecran suspect. Verification bloquee.",
        level="warning",
        hold_seconds=1.0,
    )


def signal_error():
    """
    Affiche un message d'erreur systeme.
    """
    _display_status(
        "ERREUR SYSTEME",
        "Une erreur est survenue. Verifiez la camera, le reseau ou le serveur puis reessayez.",
        level="error",
        hold_seconds=2.0,
    )


def signal_ready():
    """
    Affiche que le systeme est pret a recevoir un utilisateur.
    """
    _display_status(
        "SYSTEME PRET",
        "La borne est operationnelle. Placez-vous devant la camera pour demarrer la verification.",
        hold_seconds=0.4,
    )