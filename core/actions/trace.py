"""
core/actions/trace.py
----------------------
Action POST /trace — Segmentation 3D par texte.
Essaie LangSAM si disponible, sinon fallback géométrique par mots-clés.

Extrait de ge_gsplat_server.py → action_trace(), _try_trace_langsam(), _trace_geometric().
"""

import traceback
import numpy as np

from core.scene import get_scene, set_mask, is_scene_loaded
from core.utils.sh_color import resolve_colors_rgb


def action_trace(prompt: str, threshold: float = 0.5, colmap_dir=None) -> dict:
    """
    Segmente la scène par texte.
    Essaie LangSAM → fallback géométrique si LangSAM échoue.

    Args:
        prompt: description de l'objet à sélectionner (ex: "sofa", "les pieds")
        threshold: seuil de vote [0,1] pour l'agrégation des masques 2D
        colmap_dir: dossier COLMAP optionnel (non utilisé actuellement)

    Returns:
        dict : ok, n_selected, prompt, error
    """
    if not is_scene_loaded():
        return {
            "ok":    False,
            "error": "Aucune scène chargée. "
                     "Importez un PLY avant de segmenter.",
        }

    # Essayer LangSAM en priorité
    result = _try_trace_langsam(prompt, threshold)
    if result.get("ok"):
        return result

    # Fallback géométrique
    return _trace_geometric(prompt, threshold)


# ---------------------------------------------------------------------------
# LangSAM
# ---------------------------------------------------------------------------

def _try_trace_langsam(prompt: str, threshold: float = 0.5) -> dict:
    """Segmentation sémantique via LangSAM (GroundingDINO + SAM)."""
    try:
        from core.models import get_langsam
        from core.utils.ortho_render import render_ortho_views, aggregate_masks_3d

        scene  = get_scene()
        xyz    = scene["xyz"]
        colors = resolve_colors_rgb(colors_sh=scene.get("colors_sh"))

        # Sous-échantillonner si trop de gaussiens
        MAX_RENDER = 200_000
        if len(xyz) > MAX_RENDER:
            sel_idx       = np.random.choice(len(xyz), MAX_RENDER, replace=False)
            xyz_render    = xyz[sel_idx]
            colors_render = colors[sel_idx]
        else:
            sel_idx       = np.arange(len(xyz))
            xyz_render    = xyz
            colors_render = colors

        views = render_ortho_views(xyz_render, colors_render)
        model = get_langsam()
        views_masks = []

        for img, cam_info in views:
            try:
                results = model.predict([img], [prompt])
            except Exception as e:
                print(f"[LangSAM] Erreur vue {cam_info['name']}: {e}")
                views_masks.append((
                    np.zeros((img.height, img.width), dtype=bool), cam_info
                ))
                continue

            if results and len(results[0].get("masks", [])) > 0:
                combined = np.zeros((img.height, img.width), dtype=bool)
                for m in results[0]["masks"]:
                    try:
                        combined |= np.array(m) > 0
                    except Exception:
                        pass

                # Rejeter les masques qui couvrent trop (> 80%)
                if combined.mean() > 0.80:
                    print(f"[LangSAM] Masque trop large — ignoré")
                    combined = np.zeros_like(combined)

                views_masks.append((combined, cam_info))
            else:
                views_masks.append((
                    np.zeros((img.height, img.width), dtype=bool), cam_info
                ))

        # Vérifier qu'au moins une vue a détecté quelque chose
        if np.mean([m.mean() for m, _ in views_masks]) < 0.01:
            return {"ok": False, "error": "Aucune détection LangSAM"}

        mask  = aggregate_masks_3d(xyz, views_masks, threshold)
        n_sel = int(mask.sum())
        pct   = 100.0 * n_sel / max(scene["n"], 1)

        if pct > 80.0:
            return {"ok": False, "error": "Sélection LangSAM trop large"}

        set_mask(mask)
        print(f"[LangSAM] '{prompt}' → {n_sel:,} gaussiens ({pct:.1f}%)")
        return {"ok": True, "n_selected": n_sel, "prompt": prompt}

    except ImportError:
        return {"ok": False, "error": "lang_sam non installé"}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Fallback géométrique
# ---------------------------------------------------------------------------

def _trace_geometric(prompt: str, threshold: float = 0.5) -> dict:
    """Fallback basé sur mots-clés directionnels dans le prompt."""
    try:
        scene  = get_scene()
        xyz    = scene["xyz"]
        center = xyz.mean(axis=0)
        dists  = np.linalg.norm(xyz - center, axis=1)
        radius = np.percentile(dists, 50)
        p      = prompt.lower()

        if any(w in p for w in ["ground", "sol", "floor", "road", "rue"]):
            mask = xyz[:, 1] < center[1]
        elif any(w in p for w in ["sky", "ciel", "cloud", "nuage", "top", "haut"]):
            mask = xyz[:, 1] > center[1]
        elif any(w in p for w in ["left", "gauche"]):
            mask = xyz[:, 0] < center[0]
        elif any(w in p for w in ["right", "droite"]):
            mask = xyz[:, 0] > center[0]
        elif any(w in p for w in ["front", "avant"]):
            mask = xyz[:, 2] > center[2]
        elif any(w in p for w in ["back", "arriere", "fond"]):
            mask = xyz[:, 2] < center[2]
        else:
            mask = dists < (radius * float(threshold) * 2)

        set_mask(mask)
        n_sel = int(mask.sum())
        pct   = 100.0 * n_sel / max(scene["n"], 1)
        print(f"[Trace-Geo] '{prompt}' → {n_sel:,} gaussiens ({pct:.1f}%)")
        return {"ok": True, "n_selected": n_sel, "prompt": prompt}

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}