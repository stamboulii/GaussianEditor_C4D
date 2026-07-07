"""
core/actions/edit.py
---------------------
Action POST /edit — Édition 3D guidée par texte via InstructPix2Pix.

Extrait de ge_gsplat_server.py → action_edit() + _edit_fallback().
"""

import traceback
import numpy as np

from core.scene import (
    get_scene, get_mask, push_history,
    save_scene, sync_raw_from_scene, update_scene_fields,
)
from core.utils.sh_color import resolve_colors_rgb, rgb_to_sh
from core.utils.ortho_render import render_ortho_views
from core.gs_config import get_out_path


def action_edit(
    prompt: str,
    n_views: int        = 3,
    n_steps: int        = 20,
    n_iter: int         = 200,
    guidance_scale: float = 7.5,
    image_guidance: float = 1.5,
) -> dict:
    """
    Édition 3D guidée par texte via SDS multi-vues ou fallback IP2P.

    Args:
        prompt: instruction d'édition (ex: "make it red", "add snow")
        n_views: nombre de vues orthographiques
        n_steps: steps de diffusion
        n_iter: itérations d'optimisation
        guidance_scale: force du guidage texte
        image_guidance: force du guidage image

    Returns:
        dict : ok, ply_path, n_affected, elapsed, error
    """
    scene = get_scene()
    mask  = get_mask()

    if not scene:
        return {"ok": False, "error": "Aucune scène chargée."}
    if not prompt:
        return {"ok": False, "error": "Prompt manquant."}

    edit_mask = mask if mask is not None else np.ones(scene["n"], dtype=bool)

    if int(edit_mask.sum()) == 0:
        return {"ok": False, "error": "Aucun gaussien sélectionné. Faites d'abord /trace."}

    push_history()

    # Essayer sds_optimizer si disponible
    try:
        import sys, os
        _core_dir = os.path.dirname(os.path.abspath(__file__))
        for _p in [_core_dir, os.path.normpath(os.path.join(_core_dir, "..", "core"))]:
            if os.path.isdir(_p) and _p not in sys.path:
                sys.path.insert(0, _p)

        from sds_optimizer import run_sds
        result = run_sds(
            scene          = scene,
            mask           = edit_mask,
            prompt         = prompt,
            n_views        = n_views,
            n_steps        = n_steps,
            n_iter         = n_iter,
            lr             = 0.02,
            guidance_scale = guidance_scale,
            image_guidance = image_guidance,
            log_fn         = print,
        )
        if not result.get("ok"):
            return result

        sync_raw_from_scene()
        out_path = get_out_path("edited.ply")
        save_scene(out_path)
        return {
            "ok":        True,
            "action":    "edit",
            "ply_path":  out_path,
            "n_affected": result.get("n_affected", 0),
            "elapsed":   result.get("elapsed", 0),
        }

    except ImportError:
        return _edit_fallback_ip2p(prompt, edit_mask)
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


def _edit_fallback_ip2p(prompt: str, edit_mask: np.ndarray) -> dict:
    """
    Fallback back-projection simple via InstructPix2Pix sur 3 vues ortho.
    Utilisé si sds_optimizer.py n'est pas disponible.
    """
    try:
        import torch
        from core.models import get_ip2p

        scene  = get_scene()
        xyz    = scene["xyz"]
        colors = resolve_colors_rgb(colors_sh=scene.get("colors_sh"), n_fallback=len(xyz))

        pipe  = get_ip2p()
        views = render_ortho_views(xyz, colors)

        color_sum  = np.zeros_like(colors)
        weight_sum = np.zeros(len(xyz))

        for ref_img, cam_info in views:
            with torch.no_grad():
                result_img = pipe(
                    prompt, image=ref_img,
                    num_inference_steps    = 20,
                    image_guidance_scale   = 1.5,
                    guidance_scale         = 7.5,
                ).images[0]

            axes = cam_info["axes"]
            mn   = cam_info["mn"]
            mx   = cam_info["mx"]
            size = cam_info["size"]
            span = mx - mn
            span[span < 1e-8] = 1.0

            edited = np.array(result_img.resize((size, size))) / 255.0
            pts    = xyz[:, axes]
            pxs    = np.clip(((pts[:,0]-mn[0])/span[0]*(size-1)).astype(int), 0, size-1)
            pys    = np.clip(((pts[:,1]-mn[1])/span[1]*(size-1)).astype(int), 0, size-1)

            op = scene["opacity"]
            color_sum  += edited[pys, pxs] * op[:, None]
            weight_sum += op

        safe_w     = np.maximum(weight_sum, 1e-8)[:, None]
        new_colors = colors.copy()
        new_colors[edit_mask] = np.clip(color_sum / safe_w, 0, 1)[edit_mask]

        scene["colors_sh"] = rgb_to_sh(new_colors)
        update_scene_fields(**scene)
        sync_raw_from_scene()

        out_path = get_out_path("edited.ply")
        save_scene(out_path)

        return {
            "ok":        True,
            "action":    "edit",
            "ply_path":  out_path,
            "n_affected": int(edit_mask.sum()),
        }

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}