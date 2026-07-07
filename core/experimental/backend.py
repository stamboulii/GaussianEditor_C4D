"""
core/backend.py
---------------
Pont entre Cinema 4D et GaussianEditor.

Architecture choisie pour la Phase 2 :
  La WebUI de GaussianEditor utilise le framework "viser" (WebSocket + GUI callbacks).
  Il n'y a PAS d'API REST native. Deux stratégies sont donc implémentées :

  Stratégie A — "Patch API" (recommandée) :
    On injecte un mini serveur Flask dans le processus GaussianEditor via un fichier
    patch (ge_api_patch.py) qu'on copie dans le dossier GaussianEditor avant de
    lancer webui.py. Ce patch expose les actions (trace/edit/delete/add/save) comme
    endpoints JSON sur un port séparé (8085). C'est la méthode la plus propre.

  Stratégie B — "Ligne de commande" (fallback) :
    Pour les actions simples (edit, delete, add), GaussianEditor expose aussi des
    scripts CLI dans le dossier scripts/. On les lance directement en subprocess
    sans passer par la WebUI. Moins interactif mais plus fiable.

  Ce module implémente les deux et choisit automatiquement selon la disponibilité.

Compatible C4D 2025 (CPython 3.11). Dépendances externes : requests (optionnel).
"""

import os
import sys
import json
import time
import shutil
import signal
import subprocess
import threading
from typing import Optional, Callable, Dict, Any

# requests est optionnel (absent dans C4D natif, disponible dans l'env externe)
try:
    import requests as _requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

API_PORT      = 8085          # Port du mini serveur Flask injecté
WEBUI_PORT    = 8084          # Port de la WebUI viser de GaussianEditor
STARTUP_TIMEOUT = 180         # Secondes max pour le démarrage du serveur
POLL_INTERVAL   = 1.0         # Secondes entre chaque check de démarrage


# ---------------------------------------------------------------------------
# États du backend
# ---------------------------------------------------------------------------

class BackendState:
    STOPPED   = "stopped"
    STARTING  = "starting"
    READY     = "ready"
    BUSY      = "busy"
    ERROR     = "error"


# ---------------------------------------------------------------------------
# Contenu du patch API injecté dans GaussianEditor
# Ce code est écrit dans ge_api_patch.py côté GaussianEditor
# ---------------------------------------------------------------------------

GE_API_PATCH_CODE = '''
"""
ge_api_patch.py — Injecté par GaussianEditor C4D Plugin
Expose les actions de GaussianEditor comme API REST JSON.
Ce fichier est importé par webui.py via un monkey-patch.
"""
import threading
import json
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

_ge_instance = None   # référence à l'instance GaussianEditor
_last_result  = {}    # dernier résultat d'une action

def register_ge(ge):
    global _ge_instance
    _ge_instance = ge

class APIHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Silence les logs HTTP

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ready", "ge": _ge_instance is not None})
        elif self.path == "/result":
            self._send_json(_last_result)
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        global _last_result
        body = self._read_body()
        ge = _ge_instance

        if ge is None:
            self._send_json({"error": "GaussianEditor not initialized"}, 503)
            return

        try:
            if self.path == "/trace":
                prompt = body.get("prompt", "")
                threshold = float(body.get("threshold", 0.5))
                # Appel interne GaussianEditor
                ge.text_seg_prompt = prompt
                ge.seg_threshold   = threshold
                ge.text_seg()
                _last_result = {"ok": True, "action": "trace", "prompt": prompt}
                self._send_json(_last_result)

            elif self.path == "/edit":
                prompt = body.get("prompt", "")
                ge.edit_prompt = prompt
                ge.edit()
                out = ge.save_path if hasattr(ge, "save_path") else ""
                _last_result = {"ok": True, "action": "edit", "ply_path": out}
                self._send_json(_last_result)

            elif self.path == "/delete":
                ge.delete()
                out = ge.save_path if hasattr(ge, "save_path") else ""
                _last_result = {"ok": True, "action": "delete", "ply_path": out}
                self._send_json(_last_result)

            elif self.path == "/add":
                prompt    = body.get("prompt", "")
                mask_path = body.get("mask_path", "")
                ge.add_prompt = prompt
                if mask_path:
                    ge.inpaint_mask_path = mask_path
                ge.add()
                out = ge.save_path if hasattr(ge, "save_path") else ""
                _last_result = {"ok": True, "action": "add", "ply_path": out}
                self._send_json(_last_result)

            elif self.path == "/save":
                out_path = body.get("path", "ui_result/result.ply")
                ge.save_gaussian(out_path)
                _last_result = {"ok": True, "action": "save", "ply_path": out_path}
                self._send_json(_last_result)

            else:
                self._send_json({"error": f"Unknown endpoint: {self.path}"}, 404)

        except Exception as e:
            _last_result = {"ok": False, "error": str(e)}
            self._send_json(_last_result, 500)

def start_api_server(port=8085):
    server = HTTPServer(("127.0.0.1", port), APIHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
'''


# ---------------------------------------------------------------------------
# Classe principale : GaussianEditorBackend
# ---------------------------------------------------------------------------

class GaussianEditorBackend:
    """
    Gère le cycle de vie du processus GaussianEditor et expose ses actions.

    Utilisation typique :
        backend = GaussianEditorBackend("/path/to/GaussianEditor")
        backend.start(ply_path="scene.ply", colmap_dir="./colmap",
                      on_ready=lambda: print("Prêt !"))
        backend.edit("make it winter", on_done=lambda r: print(r))
        backend.stop()
    """

    def __init__(self, ge_root: str, api_port: int = API_PORT,
                 python_executable: str = None):
        """
        Args:
            ge_root:           Chemin vers le dossier racine de GaussianEditor
            api_port:          Port du serveur API injecté (défaut 8085)
            python_executable: Python à utiliser (défaut : détection automatique)
        """
        self.ge_root    = os.path.abspath(ge_root)
        self.api_port   = api_port
        self.api_url    = f"http://127.0.0.1:{api_port}"

        self._process:  Optional[subprocess.Popen] = None
        self._state:    str  = BackendState.STOPPED
        self._log:      list = []
        self._lock:     threading.Lock = threading.Lock()

        # Callbacks
        self._on_log:   Optional[Callable[[str], None]] = None

        # Détecter le Python externe (celui qui a PyTorch)
        self._python = python_executable or self._detect_python()

    # ------------------------------------------------------------------
    # Démarrage / arrêt
    # ------------------------------------------------------------------

    def start(self, ply_path: str, colmap_dir: str,
              on_ready: Optional[Callable] = None,
              on_error: Optional[Callable[[str], None]] = None) -> None:
        """
        Lance GaussianEditor en arrière-plan (thread non-bloquant).
        Injecte le patch API et attend que le serveur soit prêt.

        Args:
            ply_path:   Chemin absolu vers le fichier .ply
            colmap_dir: Chemin vers le dossier COLMAP
            on_ready:   Callback appelé quand le serveur est prêt
            on_error:   Callback appelé en cas d'échec
        """
        if self._state in (BackendState.STARTING, BackendState.READY, BackendState.BUSY):
            self._log_msg("Déjà démarré ou en cours de démarrage.")
            return

        self._state = BackendState.STARTING
        self._log.clear()

        def worker():
            try:
                self._inject_patch()
                self._launch_process(ply_path, colmap_dir)
                ok = self._wait_for_api()
                if ok:
                    self._state = BackendState.READY
                    self._log_msg("✓ GaussianEditor API prête.")
                    if on_ready:
                        on_ready()
                else:
                    self._state = BackendState.ERROR
                    msg = "Timeout : GaussianEditor n'a pas démarré dans le délai imparti."
                    self._log_msg(msg)
                    if on_error:
                        on_error(msg)
            except Exception as e:
                self._state = BackendState.ERROR
                self._log_msg(f"Erreur démarrage : {e}")
                if on_error:
                    on_error(str(e))

        threading.Thread(target=worker, daemon=True).start()

    def stop(self) -> None:
        """Arrête proprement le processus GaussianEditor."""
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=10)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

        self._state = BackendState.STOPPED
        self._log_msg("Backend arrêté.")

    # ------------------------------------------------------------------
    # Actions IA (toutes asynchrones via thread)
    # ------------------------------------------------------------------

    def trace(self, prompt: str, threshold: float = 0.5,
              on_done: Optional[Callable[[Dict], None]] = None,
              on_error: Optional[Callable[[str], None]] = None) -> None:
        """
        Segmentation sémantique 3D par texte (Gaussian Semantic Tracing).
        Durée estimée : 30–60 secondes.
        """
        self._run_action(
            endpoint="/trace",
            payload={"prompt": prompt, "threshold": threshold},
            on_done=on_done,
            on_error=on_error,
            label=f"trace('{prompt}')"
        )

    def edit(self, prompt: str,
             on_done: Optional[Callable[[Dict], None]] = None,
             on_error: Optional[Callable[[str], None]] = None) -> None:
        """
        Édition de textures/style via InstructPix2Pix.
        Durée estimée : 5–15 minutes.
        """
        self._run_action(
            endpoint="/edit",
            payload={"prompt": prompt},
            on_done=on_done,
            on_error=on_error,
            label=f"edit('{prompt}')"
        )

    def delete(self,
               on_done: Optional[Callable[[Dict], None]] = None,
               on_error: Optional[Callable[[str], None]] = None) -> None:
        """
        Suppression de l'objet actuellement tracé/sélectionné.
        Durée estimée : 3–10 minutes.
        """
        self._run_action(
            endpoint="/delete",
            payload={},
            on_done=on_done,
            on_error=on_error,
            label="delete()"
        )

    def add(self, prompt: str, mask_path: str = "",
            on_done: Optional[Callable[[Dict], None]] = None,
            on_error: Optional[Callable[[str], None]] = None) -> None:
        """
        Ajout d'un objet via inpainting 3D (ControlNet + image-to-3D).
        Durée estimée : 5 minutes.

        Args:
            prompt:    Texte décrivant l'objet à ajouter
            mask_path: Chemin vers le masque 2D (PNG, blanc = zone d'ajout)
        """
        self._run_action(
            endpoint="/add",
            payload={"prompt": prompt, "mask_path": mask_path},
            on_done=on_done,
            on_error=on_error,
            label=f"add('{prompt}')"
        )

    def save(self, output_path: str,
             on_done: Optional[Callable[[Dict], None]] = None,
             on_error: Optional[Callable[[str], None]] = None) -> None:
        """
        Sauvegarde la scène éditée dans un fichier .ply.

        Args:
            output_path: Chemin de sortie du fichier .ply résultat
        """
        self._run_action(
            endpoint="/save",
            payload={"path": output_path},
            on_done=on_done,
            on_error=on_error,
            label=f"save('{output_path}')"
        )

    # ------------------------------------------------------------------
    # Stratégie B : CLI fallback (sans WebUI)
    # ------------------------------------------------------------------

    def run_cli_edit(self, ply_path: str, colmap_dir: str,
                     prompt: str, output_dir: str,
                     on_done: Optional[Callable[[str], None]] = None,
                     on_error: Optional[Callable[[str], None]] = None) -> None:
        """
        Lance l'édition via les scripts CLI de GaussianEditor (sans WebUI).
        Utilise les scripts dans le dossier script/ du repo.
        C'est le fallback quand la WebUI est trop instable.

        Args:
            ply_path:   Fichier .ply source
            colmap_dir: Dossier COLMAP
            prompt:     Prompt d'édition
            output_dir: Dossier de sortie
            on_done:    Callback avec le chemin du .ply résultat
            on_error:   Callback en cas d'erreur
        """
        def worker():
            try:
                script = os.path.join(self.ge_root, "scripts", "edit.sh")
                if not os.path.isfile(script):
                    # Fallback : construire la commande Python directement
                    cmd = [
                        self._python, "launch.py",
                        "--config", "configs/gaussian-editor-edit.yaml",
                        "--train",
                        f"system.gs_source={ply_path}",
                        f"data.source={colmap_dir}",
                        f"system.prompt_processor.prompt={prompt}",
                        f"system.save_process_ply=false",
                    ]
                else:
                    cmd = ["bash", script, ply_path, colmap_dir, prompt]

                self._log_msg(f"CLI edit : {' '.join(cmd[:4])}…")

                proc = subprocess.Popen(
                    cmd,
                    cwd=self.ge_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )

                for line in iter(proc.stdout.readline, ""):
                    self._log_msg(line.rstrip())

                proc.wait()

                if proc.returncode == 0:
                    # Chercher le .ply de sortie
                    out_ply = self._find_output_ply(output_dir)
                    if on_done:
                        on_done(out_ply or output_dir)
                else:
                    if on_error:
                        on_error(f"Script CLI terminé avec code {proc.returncode}")

            except Exception as e:
                self._log_msg(f"Erreur CLI : {e}")
                if on_error:
                    on_error(str(e))

        threading.Thread(target=worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Propriétés et état
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_ready(self) -> bool:
        return self._state == BackendState.READY

    @property
    def is_busy(self) -> bool:
        return self._state == BackendState.BUSY

    @property
    def logs(self) -> list:
        return list(self._log)

    def set_log_callback(self, cb: Callable[[str], None]) -> None:
        """Callback appelé à chaque nouvelle ligne de log."""
        self._on_log = cb

    def health_check(self) -> bool:
        """Vérifie si l'API répond. Retourne True si le backend est disponible."""
        if not HAS_REQUESTS:
            return False
        try:
            r = _requests.get(f"{self.api_url}/health", timeout=3)
            return r.status_code == 200 and r.json().get("ge", False)
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Méthodes privées
    # ------------------------------------------------------------------

    def _run_action(self, endpoint: str, payload: dict,
                    on_done, on_error, label: str) -> None:
        """Exécute une action GE dans un thread séparé."""
        if self._state not in (BackendState.READY,):
            msg = f"Backend non disponible (état: {self._state})"
            self._log_msg(f"⚠ {msg}")
            if on_error:
                on_error(msg)
            return

        if not HAS_REQUESTS:
            msg = "Module 'requests' non disponible. Installez-le dans l'environnement Python."
            if on_error:
                on_error(msg)
            return

        def worker():
            with self._lock:
                self._state = BackendState.BUSY

            try:
                self._log_msg(f"→ {label}")
                r = _requests.post(
                    f"{self.api_url}{endpoint}",
                    json=payload,
                    timeout=1200  # 20 min max pour les opérations lourdes
                )
                result = r.json()
                self._log_msg(f"← {label} : {result}")

                if result.get("ok"):
                    if on_done:
                        on_done(result)
                else:
                    if on_error:
                        on_error(result.get("error", "Erreur inconnue"))

            except Exception as e:
                self._log_msg(f"✗ {label} : {e}")
                if on_error:
                    on_error(str(e))
            finally:
                with self._lock:
                    if self._state == BackendState.BUSY:
                        self._state = BackendState.READY

        threading.Thread(target=worker, daemon=True).start()

    def _inject_patch(self) -> None:
        """
        Écrit ge_api_patch.py dans le dossier GaussianEditor.
        Ce fichier sera importé au démarrage de webui.py via PYTHONPATH.
        """
        if not os.path.isdir(self.ge_root):
            raise FileNotFoundError(
                f"Dossier GaussianEditor introuvable : {self.ge_root}\n"
                "Vérifiez le chemin dans les préférences du plugin."
            )

        patch_path = os.path.join(self.ge_root, "ge_api_patch.py")
        with open(patch_path, "w", encoding="utf-8") as f:
            f.write(GE_API_PATCH_CODE)
        self._log_msg(f"Patch API injecté : {patch_path}")

    def _launch_process(self, ply_path: str, colmap_dir: str) -> None:
        """Lance le processus webui.py de GaussianEditor."""
        env = os.environ.copy()

        # Injecter le patch via sitecustomize ou PYTHONSTARTUP
        env["PYTHONSTARTUP"] = os.path.join(self.ge_root, "ge_api_patch.py")
        env["GE_API_PORT"]   = str(self.api_port)

        cmd = [
            self._python, "webui.py",
            "--gs_source",  ply_path,
            "--colmap_dir", colmap_dir,
        ]

        self._log_msg(f"Lancement : {' '.join(cmd)}")
        self._log_msg(f"Dossier   : {self.ge_root}")

        self._process = subprocess.Popen(
            cmd,
            cwd=self.ge_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Lire les logs du processus dans un thread dédié
        def read_logs():
            for line in iter(self._process.stdout.readline, ""):
                self._log_msg(line.rstrip())

        threading.Thread(target=read_logs, daemon=True).start()

    def _wait_for_api(self) -> bool:
        """
        Attend que le serveur API soit prêt.
        Retourne True si prêt dans le délai, False sinon.
        """
        if not HAS_REQUESTS:
            # Sans requests, on attend juste un délai fixe
            self._log_msg("Module 'requests' absent — attente de 30s...")
            time.sleep(30)
            return True

        self._log_msg(f"Attente du serveur API sur port {self.api_port}...")
        deadline = time.time() + STARTUP_TIMEOUT

        while time.time() < deadline:
            # Vérifier que le processus est encore vivant
            if self._process and self._process.poll() is not None:
                self._log_msg(f"Processus terminé prématurément (code {self._process.poll()})")
                return False

            try:
                r = _requests.get(f"{self.api_url}/health", timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    self._log_msg(f"API health: {data}")
                    return True
            except Exception:
                pass

            time.sleep(POLL_INTERVAL)

        return False

    def _detect_python(self) -> str:
        """
        Détecte l'exécutable Python externe qui contient PyTorch.
        Cherche dans les environnements conda/venv courants.
        """
        candidates = [
            # Environnement conda gaussianeditor
            os.path.expanduser("~/anaconda3/envs/gaussianeditor/bin/python"),
            os.path.expanduser("~/miniconda3/envs/gaussianeditor/bin/python"),
            # Windows conda
            os.path.expanduser("~/anaconda3/envs/gaussianeditor/python.exe"),
            os.path.expanduser("~/miniconda3/envs/gaussianeditor/python.exe"),
            # Python système
            "python3",
            "python",
        ]

        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        # Dernier recours : python du PATH
        return sys.executable

    def _find_output_ply(self, search_dir: str) -> Optional[str]:
        """Cherche le fichier .ply le plus récent dans un dossier."""
        if not os.path.isdir(search_dir):
            return None

        plys = []
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".ply"):
                    full = os.path.join(root, f)
                    plys.append((os.path.getmtime(full), full))

        if not plys:
            return None

        plys.sort(reverse=True)
        return plys[0][1]

    def _log_msg(self, msg: str) -> None:
        """Ajoute un message au log interne et appelle le callback si défini."""
        self._log.append(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass