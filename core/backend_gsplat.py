"""
core/backend_gsplat.py
-----------------------
Backend Phase 3 — lance core/server.py via wsl.exe depuis C4D Windows.

Le serveur tourne dans WSL2 (conda gs_c4d) et communique
avec C4D via HTTP JSON sur port 8086.
"""

import os
import sys
import subprocess
import threading
import time

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    import urllib.request
    import json as _json

_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_DIR = os.path.dirname(_THIS_DIR)

SERVER_URL  = "http://127.0.0.1:8086"
HEALTH_URL  = f"{SERVER_URL}/health"


def _win_to_wsl(path: str) -> str:
    """Convertit C:\\... -> /mnt/c/..."""
    path = path.replace("\\", "/")
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        rest  = path[2:].lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return path


def _wsl_to_win(path: str) -> str:
    """Convertit /mnt/c/... -> C:\\..."""
    if not path or not path.startswith("/mnt/"):
        return path
    parts = path[5:].split("/", 1)
    drive = parts[0].upper()
    rest  = parts[1].replace("/", "\\") if len(parts) > 1 else ""
    return f"{drive}:\\{rest}"


def _is_server_running() -> bool:
    """Verifie si le serveur repond sur port 8086."""
    try:
        if HAS_REQUESTS:
            r = requests.get(HEALTH_URL, timeout=2)
            return r.status_code == 200
        else:
            r = urllib.request.urlopen(HEALTH_URL, timeout=2)
            return r.status == 200
    except Exception:
        return False


def _post(endpoint, data, timeout=900):
    """Appel HTTP POST synchrone."""
    import json
    if HAS_REQUESTS:
        r = requests.post(f"{SERVER_URL}{endpoint}", json=data, timeout=timeout)
        return r.json()
    else:
        body = json.dumps(data).encode()
        req  = urllib.request.Request(
            f"{SERVER_URL}{endpoint}", data=body,
            headers={"Content-Type": "application/json"}
        )
        r = urllib.request.urlopen(req, timeout=timeout)
        return json.loads(r.read())


class GsplatBackend:

    def __init__(self, python_executable: str):
        self.python_executable = python_executable
        self._process          = None
        self._log_callback     = None
        self.is_ready          = False

    def set_log_callback(self, fn):
        self._log_callback = fn

    def _log(self, msg: str):
        print(msg)
        if self._log_callback:
            self._log_callback(msg)

    def start(self, ply_path: str = "", colmap_dir: str = "",
              on_ready=None, on_error=None):

        def _run():
            try:
                # -------------------------------------------------------
                # Si le serveur tourne deja
                # -------------------------------------------------------
                if _is_server_running():
                    self._log("[Backend] Serveur deja actif sur port 8086")
                    self.is_ready = True
                    if on_ready:
                        on_ready()
                    if ply_path:
                        self._auto_load(ply_path)
                    return

                # -------------------------------------------------------
                # Lancer le serveur via wsl.exe
                # -------------------------------------------------------
                plugin_wsl = _win_to_wsl(_PLUGIN_DIR)
                python_wsl = _win_to_wsl(self.python_executable)

                env_prefix = ""
                if ply_path:
                    env_prefix += f"GS_PLY_PATH='{_win_to_wsl(ply_path)}' "
                if colmap_dir:
                    env_prefix += f"GS_COLMAP_DIR='{_win_to_wsl(colmap_dir)}' "

                wsl_cmd = (
                    f"conda activate gs_c4d && "
                    f"cd '{plugin_wsl}' && "
                    f"{env_prefix}"
                    f"python -m core.server"
                )

                self._log("[Backend] Demarrage via wsl.exe...")

                self._process = subprocess.Popen(
                    ["wsl.exe", "bash", "-ic", wsl_cmd],
                    stdout  = subprocess.PIPE,
                    stderr  = subprocess.STDOUT,
                    text    = True,
                    bufsize = 1,
                )

                # Attendre que le serveur soit pret (60s max)
                for _ in range(120):
                    time.sleep(0.5)
                    if _is_server_running():
                        self.is_ready = True
                        self._log("[Backend] Serveur pret")
                        if on_ready:
                            on_ready()
                        break
                    if self._process.poll() is not None:
                        out = self._process.stdout.read()
                        raise RuntimeError(f"Serveur arrete prematurement :\n{out}")
                else:
                    raise TimeoutError(
                        "Le serveur n'a pas demarre en 60 secondes.\n"
                        "Verifiez que conda gs_c4d est installe dans WSL2."
                    )

                # Lire les logs du serveur en continu
                for line in self._process.stdout:
                    self._log(line.rstrip())

            except Exception as e:
                self.is_ready = False
                self._log(f"[Backend] ERREUR : {e}")
                if on_error:
                    on_error(str(e))

        threading.Thread(target=_run, daemon=True).start()

    def _auto_load(self, ply_path: str):
        """Charge automatiquement le PLY dans le serveur apres connexion."""
        def _load():
            try:
                wsl = _win_to_wsl(ply_path)
                result = _post("/load", {"ply_path": wsl}, timeout=30)
                if result.get("ok"):
                    self._log(f"[Backend] Scene chargee : {result.get('n', 0):,} gaussiens")
            except Exception as e:
                self._log(f"[Backend] Chargement auto echoue : {e}")
        threading.Thread(target=_load, daemon=True).start()

    def stop(self):
        self.is_ready = False
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None
            self._log("[Backend] Serveur arrete")

    # ------------------------------------------------------------------
    # Methodes HTTP
    # ------------------------------------------------------------------

    def _post_async(self, endpoint, data, on_done, on_error, timeout=900):
        def _run():
            try:
                result = _post(endpoint, data, timeout=timeout)
                # Convertir les chemins WSL -> Windows dans la reponse
                if "ply_path" in result:
                    result["ply_path"] = _wsl_to_win(result["ply_path"])
                if on_done:
                    on_done(result)
            except Exception as e:
                if on_error:
                    on_error(str(e))
        threading.Thread(target=_run, daemon=True).start()

    def generate(self, image_path, num_gaussians=0, on_done=None, on_error=None):
        # timeout=3600 : TripoSplat peut prendre jusqu'a 30 min sur petite GPU
        self._post_async("/generate",
            {"image_path": image_path, "num_gaussians": num_gaussians},
            on_done, on_error, timeout=3600)

    def generate_and_add(self, image_path, num_gaussians=0, on_done=None, on_error=None):
        self._post_async("/generate_and_add",
            {"image_path": image_path, "num_gaussians": num_gaussians},
            on_done, on_error, timeout=3600)

    def load(self, ply_path, on_done=None, on_error=None):
        self._post_async("/load", {"ply_path": ply_path}, on_done, on_error)

    def trace(self, prompt, threshold=0.5, on_done=None, on_error=None):
        self._post_async("/trace",
            {"prompt": prompt, "threshold": threshold},
            on_done, on_error, timeout=120)

    def edit(self, prompt, on_done=None, on_error=None):
        self._post_async("/edit", {"prompt": prompt}, on_done, on_error, timeout=600)

    def delete(self, on_done=None, on_error=None):
        self._post_async("/delete", {}, on_done, on_error)

    def crop(self, on_done=None, on_error=None):
        self._post_async("/crop", {}, on_done, on_error)

    def undo(self, on_done=None, on_error=None):
        self._post_async("/undo", {}, on_done, on_error)

    def opacity(self, value, on_done=None, on_error=None):
        self._post_async("/opacity", {"value": value}, on_done, on_error)

    def scale(self, factor, on_done=None, on_error=None):
        self._post_async("/scale", {"factor": factor}, on_done, on_error)

    def save(self, path="", on_done=None, on_error=None):
        self._post_async("/save", {"path": path}, on_done, on_error)

    def export_splat(self, path="", on_done=None, on_error=None):
        self._post_async("/export_splat", {"path": path}, on_done, on_error)