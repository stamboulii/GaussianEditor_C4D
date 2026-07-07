"""
core/actions/load.py
---------------------
Action POST /load — Charger un fichier PLY existant dans la scène.
"""

import os
from core.scene import load_scene, get_scene


def action_load(ply_path: str) -> dict:
    """
    Charge un fichier PLY Gaussian Splatting dans la scène courante.

    Args:
        ply_path: chemin absolu vers le fichier .ply

    Returns:
        dict : ok, n, error
    """
    if not ply_path:
        return {"ok": False, "error": "ply_path manquant"}

    if not os.path.isfile(ply_path):
        return {"ok": False, "error": f"Fichier introuvable : {ply_path}"}

    ok = load_scene(ply_path)
    if not ok:
        return {"ok": False, "error": f"Impossible de charger {os.path.basename(ply_path)}"}

    return {
        "ok": True,
        "n":  get_scene().get("n", 0),
    }