"""
core/actions/adjustments.py
----------------------------
Actions POST /opacity, /scale, /save, /export_splat.
"""

import os
import traceback
import numpy as np

from core.scene import (
    get_scene, get_mask, push_history,
    save_scene, sync_raw_from_scene, update_scene_fields,
)
from core.gs_config import get_out_path


def action_opacity(value: float = 0.3) -> dict:
    """
    Modifie l'opacité des gaussiens sélectionnés par /trace.

    Args:
        value: opacité cible [0.001, 0.999]
    """
    scene = get_scene()
    mask  = get_mask()

    if mask is None:
        return {"ok": False, "error": "Faites d'abord un /trace."}

    try:
        push_history()

        pidx = {name: i for i, name in enumerate(scene["props"])}
        if "opacity" not in pidx:
            return {"ok": False, "error": "Ce PLY n'a pas d'attribut opacity."}

        value     = float(np.clip(value, 0.001, 0.999))
        new_logit = float(np.log(value / (1.0 - value)))

        scene["raw"][mask, pidx["opacity"]] = new_logit
        scene["opacity"] = 1.0 / (1.0 + np.exp(-scene["raw"][:, pidx["opacity"]]))
        update_scene_fields(**scene)

        n_sel    = int(mask.sum())
        out_path = get_out_path("opacity.ply")
        save_scene(out_path)

        print(f"[Opacity] {n_sel:,} gaussiens → opacity={value:.2f}")
        return {
            "ok":        True,
            "action":    "opacity",
            "ply_path":  out_path,
            "n_affected": n_sel,
        }

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


def action_scale(factor: float = 2.0) -> dict:
    """
    Modifie la taille des gaussiens sélectionnés par /trace.

    Args:
        factor: facteur multiplicatif (ex: 2.0 = doubler la taille)
    """
    scene = get_scene()
    mask  = get_mask()

    if mask is None:
        return {"ok": False, "error": "Faites d'abord un /trace."}

    try:
        push_history()

        pidx       = {name: i for i, name in enumerate(scene["props"])}
        scale_keys = [k for k in ["scale_0", "scale_1", "scale_2"] if k in pidx]
        if not scale_keys:
            return {"ok": False, "error": "Ce PLY n'a pas d'attribut scale."}

        log_factor = float(np.log(max(0.001, float(factor))))
        for k in scale_keys:
            scene["raw"][mask, pidx[k]] += log_factor

        scene["scales"] = np.exp(
            scene["raw"][:, [pidx[k] for k in scale_keys]]
        )
        update_scene_fields(**scene)

        n_sel    = int(mask.sum())
        out_path = get_out_path("scaled.ply")
        save_scene(out_path)

        print(f"[Scale] {n_sel:,} gaussiens → factor={factor:.2f}")
        return {
            "ok":        True,
            "action":    "scale",
            "ply_path":  out_path,
            "n_affected": n_sel,
        }

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


def action_save(out_path: str = "") -> dict:
    """
    Sauvegarde la scène courante.

    Args:
        out_path: chemin de sortie. Si vide, utilise get_out_path("result.ply")
    """
    if not out_path:
        out_path = get_out_path("result.ply")

    ok = save_scene(out_path)
    return {
        "ok":       ok,
        "action":   "save",
        "ply_path": os.path.abspath(out_path),
    }


def action_export_splat(out_path: str = "") -> dict:
    """
    Exporte la scène au format .splat (léger, compatible viewers web).
    Utilise la méthode to_splat_bytes() de la classe Gaussian de TripoSplat.

    Args:
        out_path: chemin de sortie .splat. Si vide, utilise get_out_path("export.splat")
    """
    scene = get_scene()
    if not scene:
        return {"ok": False, "error": "Aucune scène chargée."}

    if not out_path:
        out_path = get_out_path("export.splat")

    try:
        # Reconstruire un objet Gaussian TripoSplat depuis les données de la scène
        import sys
        from core.gs_config import TRIPOSPLAT_DIR
        if TRIPOSPLAT_DIR not in sys.path:
            sys.path.insert(0, TRIPOSPLAT_DIR)

        from triposplat import Gaussian
        import torch

        n   = scene["n"]
        xyz = torch.tensor(scene["xyz"], dtype=torch.float32)

        # Calculer l'AABB de la scène
        mn  = xyz.min(0).values
        mx  = xyz.max(0).values
        aabb = torch.cat([mn, mx - mn]).tolist()

        gs = Gaussian(aabb=aabb, device="cpu")

        # Renseigner les attributs
        gs._xyz       = ((xyz - mn) / (mx - mn + 1e-8) - 0.5).unsqueeze(0)
        gs._features_dc = torch.tensor(
            scene["colors_sh"].reshape(n, 1, 3), dtype=torch.float32
        )
        gs._opacity   = torch.tensor(scene["opacity"].reshape(n, 1), dtype=torch.float32)
        gs._scaling   = torch.log(torch.tensor(scene["scales"], dtype=torch.float32))
        gs._rotation  = torch.tensor(scene["rotations"] - gs.rots_bias.cpu().numpy(), dtype=torch.float32)

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        gs.save_splat(out_path)

        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"[ExportSplat] {out_path} ({size_mb:.1f} MB)")

        return {
            "ok":       True,
            "action":   "export_splat",
            "ply_path": os.path.abspath(out_path),
            "size_mb":  round(size_mb, 2),
            "n":        n,
        }

    except ImportError:
        return {
            "ok":    False,
            "error": "TripoSplat non disponible — export .splat impossible. "
                     "Vérifiez TRIPOSPLAT_DIR dans gs_config.py",
        }
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}