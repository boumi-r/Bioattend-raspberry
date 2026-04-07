# src/camera.py

from picamera2 import Picamera2
import cv2
import time
import logging
import threading
import numpy as np
import config

logger = logging.getLogger(__name__)

# ── Couleurs (BGR) ────────────────────────────────────────────
COLOR_GREEN  = (0,   220,  0)
COLOR_RED    = (0,   0,   220)
COLOR_ORANGE = (0,   165, 255)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0,   0,   0)
COLOR_BLUE   = (255, 120,  0)

WINDOW_NAME  = "BioAttend — Vérification"


class CameraManager:
    def __init__(self):
        self.camera       = None
        self._display_on  = False          # fenêtre active ?
        self._status_text = "En attente..."
        self._status_color = COLOR_WHITE
        self._bbox        = None           # [x, y, w, h] visage détecté
        self._frame_lock  = threading.Lock()
        self._last_frame  = None           # dernier frame capturé (BGR)
        self._preview_thread = None
        self._stop_preview   = threading.Event()

    # ══════════════════════════════════════════════════════════
    # OUVERTURE / FERMETURE
    # ══════════════════════════════════════════════════════════

    def open(self):
        """Initialise la caméra PiCamera2 et ouvre la fenêtre d'affichage."""
        try:
            self.camera = Picamera2()

            config_cam = self.camera.create_preview_configuration(
                main={
                    "size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
                    "format": "RGB888"
                }
            )
            self.camera.configure(config_cam)
            self.camera.start()

            logger.info(f"Caméra démarrée — chauffe {config.CAMERA_WARMUP}s...")
            time.sleep(config.CAMERA_WARMUP)
            logger.info("Caméra prête")

            # Démarrer l'affichage en temps réel
            self._start_display()

        except Exception as e:
            logger.error(f"Erreur caméra : {e}")
            raise

    def close(self):
        """Ferme la caméra et la fenêtre d'affichage."""
        self._stop_display()
        if self.camera:
            self.camera.stop()
            self.camera.close()
            logger.info("Caméra fermée")

    # ══════════════════════════════════════════════════════════
    # CAPTURE
    # ══════════════════════════════════════════════════════════

    def capture_frame(self) -> np.ndarray:
        """Capture une image — numpy array BGR pour OpenCV."""
        frame = self.camera.capture_array()
        bgr   = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Mettre à jour le dernier frame pour l'affichage
        with self._frame_lock:
            self._last_frame = bgr.copy()
        return bgr

    def capture_jpeg(self) -> bytes:
        """Capture une image et retourne bytes JPEG."""
        frame = self.capture_frame()
        success, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not success:
            raise RuntimeError("Erreur encodage JPEG")
        return buffer.tobytes()

    def capture_image(self) -> bytes:
        """Alias pour capture_jpeg() — compatible avec main.py."""
        return self.capture_jpeg()

    # ══════════════════════════════════════════════════════════
    # MISE À JOUR AFFICHAGE (appelé depuis main.py)
    # ══════════════════════════════════════════════════════════

    def set_status(self, text: str, color: tuple = COLOR_WHITE):
        """
        Met à jour le texte de statut affiché sur l'écran.

        Exemples d'appel depuis main.py :
            cam.set_status("Analyse en cours...", COLOR_ORANGE)
            cam.set_status("Accès autorisé ✓",   COLOR_GREEN)
            cam.set_status("Accès refusé ✗",     COLOR_RED)
            cam.set_status("En attente...")
        """
        self._status_text  = text
        self._status_color = color

    def set_bbox(self, bbox: list | None):
        """
        Affiche un rectangle autour du visage détecté.
        bbox = [x, y, w, h] ou None pour effacer
        """
        self._bbox = bbox

    # ══════════════════════════════════════════════════════════
    # THREAD D'AFFICHAGE
    # ══════════════════════════════════════════════════════════

    def _start_display(self):
        """Lance le thread de preview en temps réel."""
        self._display_on = True
        self._stop_preview.clear()
        self._preview_thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
            name="CameraDisplay"
        )
        self._preview_thread.start()
        logger.info("Affichage temps réel démarré")

    def _stop_display(self):
        """Arrête le thread de preview."""
        self._stop_preview.set()
        if self._preview_thread:
            self._preview_thread.join(timeout=2)
        cv2.destroyAllWindows()
        logger.info("Affichage temps réel arrêté")

    def _display_loop(self):
        """
        Boucle principale du thread d'affichage.
        Tourne à ~15 FPS — suffisant pour du temps réel sur Pi.
        """
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN       # plein écran sur le Pi
        )

        while not self._stop_preview.is_set():
            # ── Lire le dernier frame ────────────────────────
            with self._frame_lock:
                frame = self._last_frame.copy() if self._last_frame is not None else None

            if frame is None:
                # Pas encore de frame — afficher un écran d'attente
                frame = _make_waiting_screen(config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
            else:
                frame = self._draw_overlay(frame)

            cv2.imshow(WINDOW_NAME, frame)

            # Quitter si on appuie sur 'q'
            if cv2.waitKey(66) & 0xFF == ord('q'):   # ~15 FPS
                break

        cv2.destroyAllWindows()

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Dessine sur le frame :
          - Rectangle autour du visage (si bbox définie)
          - Bandeau de statut en bas
          - Logo / titre en haut
        """
        h, w = frame.shape[:2]

        # ── Bounding box visage ──────────────────────────────
        if self._bbox is not None:
            x, y, bw, bh = self._bbox
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), COLOR_GREEN, 2)
            cv2.putText(
                frame, "Visage détecté",
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2
            )

        # ── Bandeau titre (haut) ─────────────────────────────
        cv2.rectangle(frame, (0, 0), (w, 50), COLOR_BLACK, -1)
        cv2.putText(
            frame, "BioAttend — Système de pointage biométrique",
            (15, 33),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BLUE, 2
        )

        # ── Bandeau statut (bas) ─────────────────────────────
        cv2.rectangle(frame, (0, h - 60), (w, h), COLOR_BLACK, -1)

        # Fond coloré derrière le texte de statut
        text_size = cv2.getTextSize(
            self._status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2
        )[0]
        cv2.rectangle(
            frame,
            (10, h - 55),
            (20 + text_size[0], h - 10),
            COLOR_BLACK, -1
        )
        cv2.putText(
            frame, self._status_text,
            (15, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, self._status_color, 2
        )

        # ── Heure en bas à droite ────────────────────────────
        heure = time.strftime("%H:%M:%S")
        cv2.putText(
            frame, heure,
            (w - 110, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 1
        )

        return frame


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════

def _make_waiting_screen(w: int, h: int) -> np.ndarray:
    """Génère un écran noir avec message d'attente."""
    screen = np.zeros((h, w, 3), dtype=np.uint8)
    text   = "Initialisation de la caméra..."
    size   = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    x      = (w - size[0]) // 2
    y      = (h - size[1]) // 2
    cv2.putText(screen, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_WHITE, 2)
    return screen


# ── Export des couleurs pour main.py ─────────────────────────
__all__ = [
    "CameraManager",
    "COLOR_GREEN", "COLOR_RED", "COLOR_ORANGE",
    "COLOR_WHITE", "COLOR_BLACK", "COLOR_BLUE",
]