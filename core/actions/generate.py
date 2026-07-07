"""
core/actions/generate.py
-------------------------
Action POST /generate       — Image → TripoSplat → nouvelle scène.
Action POST /generate_and_add — Image → TripoSplat → PLY séparé (pas fusionné).

Fix juillet 2026 :
  - generate_and_add retourne le PLY du nouvel objet SANS fusionner
  - La fusion se fait uniquement au /save depuis C4D
  - Permet de déplacer/redimensionner chaque objet indépendamment dans C4D
"""

import os
import traceback
import numpy as np

from core.gs_config import get_out_path, recommended_num_gaussians, VRAM_GB
from core.models    import generate_from_image
from core.scene     import load_scene, is_scene_loaded
from core.scene.ply_loader import load_scene as _load_into_scene


def action_generate(image_path: str, num_gaussians: int = 0) -> dict:
    """
    Génère un Gaussian Splat depuis une image et charge la scène.
    Remplace la scène existante.
    """
    if not image_path or not os.path.isfile(image_path):
        return {"ok": False, "error": f"Image introuvable : {image_path}"}

    if num_gaussians <= 0:
        num_gaussians = recommended_num_gaussians(VRAM_GB)

    base     = os.path.splitext(os.path.basename(image_path))[0]
    out_path = get_out_path(f"{base}_triposplat.ply")

    result = generate_from_image(
        image_path      = image_path,
        num_gaussians   = num_gaussians,
        output_ply_path = out_path,
        release_after   = True,
    )

    if not result["ok"]:
        return result

    ok = _load_into_scene(result["ply_path"])
    if not ok:
        return {"ok": False, "error": "PLY généré mais impossible de charger la scène"}

    from core.scene import get_scene
    return {
        "ok":       True,
        "action":   "generate",
        "ply_path": result["ply_path"],
        "n":        get_scene().get("n", 0),
    }


def action_generate_and_add(image_path: str, num_gaussians: int = 0) -> dict:
    """
    Génère un Gaussian Splat depuis une image et retourne le PLY séparé.

    NE FUSIONNE PAS avec la scène existante.
    C4D reçoit deux chemins PLY séparés et crée deux objets indépendants.
    La fusion n'arrive qu'au /save depuis C4D après repositionnement.
    """
    if not image_path or not os.path.isfile(image_path):
        return {"ok": False, "error": f"Image introuvable : {image_path}"}

    if num_gaussians <= 0:
        num_gaussians = recommended_num_gaussians(VRAM_GB)

    base     = os.path.splitext(os.path.basename(image_path))[0]
    out_path = get_out_path(f"{base}_triposplat.ply")

    result = generate_from_image(
        image_path      = image_path,
        num_gaussians   = num_gaussians,
        output_ply_path = out_path,
        release_after   = True,
    )

    if not result["ok"]:
        return result

    # Charger le nouvel objet dans le serveur comme scène courante
    # (pour que les actions /trace /edit etc. puissent travailler dessus)
    _load_into_scene(result["ply_path"])

    from core.scene import get_scene
    return {
        "ok":            True,
        "action":        "generate_and_add",
        "ply_path":      result["ply_path"],   # PLY du nouvel objet
        "n":             get_scene().get("n", 0),
        "separate":      True,                  # Signal pour C4D : objet séparé
    }