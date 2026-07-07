"""
core/scene/ply_loader.py
-------------------------
Chargement de fichiers PLY Gaussian Splatting en mémoire.

Extrait de ge_gsplat_server.py → fonction load_scene().
Utilise sh_color.py pour la conversion des couleurs.
"""

import traceback
import numpy as np

from core.scene.scene_state import set_scene
from core.utils.sh_color import rgb_to_sh


def load_scene(ply_path: str) -> bool:
    """
    Charge un fichier PLY Gaussian Splatting en mémoire.
    Gère les attributs manquants avec des valeurs par défaut.

    Args:
        ply_path: chemin absolu vers le fichier .ply

    Returns:
        True si chargement réussi, False sinon
    """
    try:
        with open(ply_path, "rb") as f:
            props, n_verts = [], 0
            while True:
                line = f.readline().decode("utf-8", errors="replace").strip()
                if line.startswith("element vertex"):
                    n_verts = int(line.split()[-1])
                elif line.startswith("property float"):
                    props.append(line.split()[-1])
                elif line == "end_header":
                    break
            n_props = len(props)
            raw     = f.read(n_verts * n_props * 4)
            data    = np.frombuffer(raw, dtype="<f4").reshape(n_verts, n_props).copy()

        idx = {name: i for i, name in enumerate(props)}

        def _get(keys, default=None):
            cols = [idx[k] for k in keys if k in idx]
            if not cols:
                return default
            return data[:, cols] if len(cols) > 1 else data[:, cols[0]]

        # XYZ obligatoire
        xyz_cols = [k for k in ["x", "y", "z"] if k in idx]
        if len(xyz_cols) < 3:
            raise ValueError(f"PLY sans coordonnées XYZ : {props[:10]}")

        # Opacité (logit-space dans le PLY)
        raw_opacity = _get(["opacity"])
        opacity = (1.0 / (1.0 + np.exp(-raw_opacity))
                   if raw_opacity is not None else np.ones(n_verts, dtype=np.float32))

        # Scales (log-space dans le PLY)
        raw_scales = _get(["scale_0", "scale_1", "scale_2"])
        scales = (np.exp(raw_scales)
                  if raw_scales is not None else np.ones((n_verts, 3), dtype=np.float32) * 0.01)

        # Rotations (quaternion)
        raw_rots = _get(["rot_0", "rot_1", "rot_2", "rot_3"])
        rotations = (raw_rots if raw_rots is not None
                     else np.tile([1, 0, 0, 0], (n_verts, 1)).astype(np.float32))

        # Couleurs SH (DC term)
        # Priorité : f_dc (SH) → RGB entier [0-255] → zéro
        raw_sh = _get(["f_dc_0", "f_dc_1", "f_dc_2"])
        if raw_sh is None:
            raw_rgb = _get(["red", "green", "blue"])
            if raw_rgb is not None:
                # Convertir RGB [0-255] en SH via sh_color
                raw_sh = rgb_to_sh(raw_rgb / 255.0)
            else:
                raw_sh = np.zeros((n_verts, 3), dtype=np.float32)

        scene = {
            "ply_path":  ply_path,
            "n":         n_verts,
            "xyz":       data[:, [idx["x"], idx["y"], idx["z"]]],
            "opacity":   opacity,
            "scales":    scales,
            "rotations": rotations,
            "colors_sh": raw_sh,
            "props":     props,
            "raw":       data,
        }

        set_scene(scene)
        print(f"[Scene] {n_verts:,} gaussiens chargés depuis {ply_path}")
        return True

    except Exception:
        traceback.print_exc()
        return False