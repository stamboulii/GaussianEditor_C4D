"""
core/server.py
--------------
Serveur HTTP local pour GaussianEditor C4D Plugin.

Remplace ge_gsplat_server.py — ce fichier ne contient QUE le routing HTTP.
Toute la logique métier est dans core/actions/, core/scene/, core/models/.

Endpoints exposés :
  GET  /health              → statut + nb gaussiens
  GET  /scene_info          → info scène courante
  POST /generate            → image → TripoSplat → nouvelle scène
  POST /generate_and_add    → image → TripoSplat → fusion scène existante
  POST /load                → charger un .ply existant
  POST /trace               → segmentation 3D par texte
  POST /edit                → édition guidée par texte (InstructPix2Pix)
  POST /delete              → suppression gaussiens sélectionnés
  POST /crop                → garder uniquement la sélection
  POST /undo                → annuler la dernière action
  POST /opacity             → modifier opacité des gaussiens sélectionnés
  POST /scale               → modifier taille des gaussiens sélectionnés
  POST /save                → sauvegarder la scène courante
  POST /export_splat        → exporter en format .splat (web)

Compatible : Python 3.10+, PyTorch 2.0+, CUDA 11.8 / 12.1
"""

import os
import sys
import io
import json
import traceback
import threading
import socketserver
from http.server import HTTPServer, BaseHTTPRequestHandler

# Force UTF-8 sur stdout/stderr (important sur Windows)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# ---------------------------------------------------------------------------
# Configuration et setup
# ---------------------------------------------------------------------------

from core.gs_config import PORT, HOST, PLY_PATH, DEVICE, VRAM_GB, setup_hf_cache
from core.scene     import load_scene, get_scene, get_mask, is_scene_loaded, history_size
from core.models    import register_all_models
from core.actions.merge_save import action_merge_and_save
from core.actions   import (
    action_generate, action_generate_and_add,
    action_load,
    action_trace,
    action_delete, action_crop, action_undo,
    action_opacity, action_scale, action_save, action_export_splat,
)

# Action edit importée séparément (dépendance optionnelle IP2P)
try:
    from core.actions.edit import action_edit
except ImportError:
    def action_edit(*args, **kwargs):
        return {"ok": False, "error": "Module edit non disponible"}


# ---------------------------------------------------------------------------
# Serveur HTTP multi-thread
# ---------------------------------------------------------------------------

class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    """
    Serveur HTTP multi-thread.
    Chaque requête est traitée dans un thread séparé.
    Évite le WinError 10053 quand le client coupe la connexion après timeout.
    """
    daemon_threads      = True
    allow_reuse_address = True


class APIHandler(BaseHTTPRequestHandler):

    def log_message(self, *args):
        pass  # Supprimer les logs HTTP verbeux

    def _send_json(self, data, status=200):
        """Envoi JSON robuste — ignore les ConnectionAbortedError."""
        try:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError, OSError):
            pass

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    # ------------------------------------------------------------------
    # GET
    # ------------------------------------------------------------------

    def do_GET(self):
        scene = get_scene()
        mask  = get_mask()

        if self.path == "/health":
            self._send_json({
                "status":          "ready",
                "n_gaussians":     scene.get("n", 0),
                "device":          DEVICE,
                "vram_gb":         round(VRAM_GB, 1),
                "has_mask":        mask is not None,
                "history_size":    history_size(),
            })

        elif self.path == "/scene_info":
            self._send_json({
                "n":           scene.get("n", 0),
                "ply_path":    scene.get("ply_path", ""),
                "has_mask":    mask is not None,
                "n_selected":  int(mask.sum()) if mask is not None else 0,
                "history_size": history_size(),
            })

        else:
            self._send_json({"error": f"Endpoint inconnu : {self.path}"}, 404)

    # ------------------------------------------------------------------
    # POST
    # ------------------------------------------------------------------

    def do_POST(self):
        body = self._read_body()
        try:
            result = self._dispatch(body)
            self._send_json(result)
        except Exception as e:
            traceback.print_exc()
            self._send_json({"ok": False, "error": str(e)}, 500)

    def _dispatch(self, body: dict) -> dict:
        """Route la requête vers la bonne action."""
        p = self.path

        if p == "/generate":
            return action_generate(
                image_path    = body.get("image_path", ""),
                num_gaussians = int(body.get("num_gaussians", 0)),
            )

        elif p == "/generate_and_add":
            return action_generate_and_add(
                image_path    = body.get("image_path", ""),
                num_gaussians = int(body.get("num_gaussians", 0)),
            )

        elif p == "/load":
            return action_load(body.get("ply_path", ""))

        elif p == "/trace":
            return action_trace(
                prompt     = body.get("prompt", ""),
                threshold  = float(body.get("threshold", 0.5)),
                colmap_dir = body.get("colmap_dir", None),
            )

        elif p == "/edit":
            return action_edit(
                prompt         = body.get("prompt", ""),
                n_views        = int(body.get("n_views", 3)),
                n_steps        = int(body.get("n_steps", 20)),
                n_iter         = int(body.get("n_iter", 200)),
                guidance_scale = float(body.get("guidance_scale", 7.5)),
                image_guidance = float(body.get("image_guidance", 1.5)),
            )

        elif p == "/delete":
            return action_delete()

        elif p == "/crop":
            return action_crop()

        elif p == "/undo":
            return action_undo()

        elif p == "/opacity":
            return action_opacity(float(body.get("value", 0.3)))

        elif p == "/scale":
            return action_scale(float(body.get("factor", 2.0)))

        elif p == "/save":
            return action_save(body.get("path", ""))

        elif p == "/merge_and_save":
            return action_merge_and_save(
                objects  = body.get("objects", []),
                out_path = body.get("out_path", ""),
            )

        elif p == "/export_splat":
            return action_export_splat(body.get("path", ""))

        else:
            return {"ok": False, "error": f"Endpoint inconnu : {self.path}"}


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------

def _preload_langsam():
    """Précharge LangSAM en arrière-plan pour éviter le timeout au premier /trace."""
    try:
        from core.models import get_langsam
        get_langsam()
        print("[Server] LangSAM prêt")
    except Exception as e:
        print(f"[Server] LangSAM non disponible ({e}) — fallback géométrique actif")


def main():
    # Setup HuggingFace cache
    setup_hf_cache()

    # Enregistrer tous les modèles (lazy — pas encore chargés)
    register_all_models()

    # Charger la scène si PLY fourni en variable d'environnement
    if PLY_PATH and os.path.isfile(PLY_PATH):
        load_scene(PLY_PATH)
    elif PLY_PATH:
        print(f"[Server] Avertissement : PLY introuvable : {PLY_PATH}")

    # Précharger LangSAM en arrière-plan
    threading.Thread(target=_preload_langsam, daemon=True).start()

    # Démarrer le serveur
    server = ThreadedHTTPServer((HOST, PORT), APIHandler)
    print(f"[Server] API prête sur http://{HOST}:{PORT}")
    print(f"[Server] Scène : {PLY_PATH or '(aucune)'}")
    print(f"[Server] Device : {DEVICE} | VRAM : {VRAM_GB:.1f}GB")
    print(f"[Server] Endpoints : /generate /generate_and_add /load /trace "
          f"/edit /delete /crop /undo /opacity /scale /save /merge_and_save /export_splat")
    server.serve_forever()


if __name__ == "__main__":
    main()