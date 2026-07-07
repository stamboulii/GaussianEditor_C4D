"""
core/scene/ply_writer.py
-------------------------
Sauvegarde de la scène courante au format PLY binaire.

Extrait de ge_gsplat_server.py → fonction save_scene().
"""

import os
import traceback
import numpy as np

from core.scene.scene_state import get_scene
from core.gs_config import get_out_path


def save_scene(out_path: str = "") -> bool:
    """
    Sauvegarde la scène courante au format PLY binaire little-endian.

    Args:
        out_path: chemin de sortie. Si vide, utilise get_out_path("result.ply")

    Returns:
        True si sauvegarde réussie, False sinon
    """
    scene = get_scene()
    if not scene:
        return False

    if not out_path:
        out_path = get_out_path("result.ply")

    try:
        props = scene["props"]
        data  = scene["raw"].copy()
        n     = scene["n"]

        header_lines = [
            "ply",
            "format binary_little_endian 1.0",
            "comment GaussianEditor C4D",
            f"element vertex {n}",
        ]
        for p in props:
            header_lines.append(f"property float {p}")
        header_lines.append("end_header\n")
        header = "\n".join(header_lines).encode("utf-8")

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(header)
            f.write(data.astype("<f4").tobytes())

        print(f"[Scene] Sauvegardée : {out_path}")
        return True

    except Exception:
        traceback.print_exc()
        return False