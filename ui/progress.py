"""
ui/progress.py
--------------
Gestion de la barre de progression pour les operations longues.

Usage :
    with ProgressContext(dialog, "Edition en cours", total_steps=20) as prog:
        for i in range(20):
            prog.step(f"Etape {i+1}/20")
            time.sleep(0.5)
"""

import threading
import time

try:
    import c4d
    INSIDE_C4D = True
except ImportError:
    INSIDE_C4D = False


class ProgressReporter:
    """
    Reporte la progression d'une operation longue vers l'UI C4D.
    Thread-safe — peut etre appele depuis n'importe quel thread.
    """

    def __init__(self, label: str = "", total: int = 100):
        self.label   = label
        self.total   = max(total, 1)
        self.current = 0
        self._lock   = threading.Lock()
        self._done   = False

    def step(self, message: str = "", increment: int = 1):
        """Avance d'un pas et met a jour le statut C4D."""
        with self._lock:
            self.current = min(self.current + increment, self.total)
            pct = int(100 * self.current / self.total)
            text = f"{self.label} {message} ({pct}%)" if message else f"{self.label} ({pct}%)"

        if INSIDE_C4D:
            try:
                c4d.gui.StatusSetText(text)
                c4d.gui.StatusSetBar(pct)
            except Exception:
                pass

    def set(self, current: int, message: str = ""):
        """Positionne la progression a une valeur precise."""
        with self._lock:
            self.current = min(max(current, 0), self.total)
            pct = int(100 * self.current / self.total)
            text = f"{self.label} {message} ({pct}%)" if message else f"{self.label} ({pct}%)"

        if INSIDE_C4D:
            try:
                c4d.gui.StatusSetText(text)
                c4d.gui.StatusSetBar(pct)
            except Exception:
                pass

    def done(self, message: str = "Termine"):
        """Marque la progression comme terminee."""
        with self._lock:
            self._done = True
        if INSIDE_C4D:
            try:
                c4d.gui.StatusSetText(f"{self.label} — {message}")
                c4d.gui.StatusSetBar(100)
            except Exception:
                pass

    def error(self, message: str):
        """Marque une erreur."""
        with self._lock:
            self._done = True
        if INSIDE_C4D:
            try:
                c4d.gui.StatusSetText(f"Erreur : {message}")
                c4d.gui.StatusSetBar(0)
            except Exception:
                pass

    def spin(self, message: str = ""):
        """Active l'animation de chargement C4D (spinner)."""
        if INSIDE_C4D:
            try:
                text = f"{self.label} {message}..." if message else f"{self.label}..."
                c4d.gui.StatusSetText(text)
                c4d.gui.StatusSetSpin()
            except Exception:
                pass


class ProgressContext:
    """
    Context manager pour la progression.

    Usage :
        with ProgressContext("Segmentation", total=10) as prog:
            for i in range(10):
                prog.step(f"Vue {i}")
    """

    def __init__(self, label: str, total: int = 100):
        self.reporter = ProgressReporter(label, total)

    def __enter__(self) -> ProgressReporter:
        self.reporter.spin()
        return self.reporter

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.reporter.error(str(exc_val))
        else:
            self.reporter.done()
        return False


class PollingProgress:
    """
    Progression par polling — interroge regulierement le serveur
    pour mettre a jour la barre de progression sans bloquer l'UI.

    Usage :
        poller = PollingProgress(
            poll_fn=lambda: backend.get_progress(),
            label="Edition",
            interval=2.0
        )
        poller.start()
        # ... operation longue ...
        poller.stop()
    """

    def __init__(self, poll_fn, label: str = "", interval: float = 2.0):
        self._poll_fn  = poll_fn
        self._label    = label
        self._interval = interval
        self._thread   = None
        self._running  = False
        self._reporter = ProgressReporter(label)

    def start(self):
        """Demarre le thread de polling."""
        self._running = True
        self._reporter.spin()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, message: str = "Termine"):
        """Arrete le polling."""
        self._running = False
        self._reporter.done(message)

    def _run(self):
        elapsed = 0
        while self._running:
            time.sleep(self._interval)
            elapsed += self._interval
            try:
                result = self._poll_fn()
                if result is not None:
                    pct  = result.get("progress", 0)
                    msg  = result.get("message", "")
                    self._reporter.set(pct, msg)
            except Exception:
                pass
            # Mise a jour du temps ecoule dans le statut
            if INSIDE_C4D and self._running:
                try:
                    mins = int(elapsed // 60)
                    secs = int(elapsed % 60)
                    c4d.gui.StatusSetText(
                        f"{self._label} ({mins:02d}:{secs:02d} ecoule)"
                    )
                    c4d.gui.StatusSetSpin()
                except Exception:
                    pass