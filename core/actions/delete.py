"""
core/actions/delete.py
-----------------------
Action POST /delete — Supprime les gaussiens sélectionnés par /trace.
Action POST /crop   — Garde uniquement les gaussiens sélectionnés.
Action POST /undo   — Restaure le dernier état sauvegardé.
"""

import traceback
import numpy as np

from core.scene import (
    get_scene, get_mask, clear_mask,
    push_history, pop_history, history_size,
    save_scene, update_scene_fields,
)
from core.gs_config import get_out_path


def action_delete() -> dict:
    """
    Supprime les gaussiens sélectionnés par /trace.
    Sauvegarde l'état précédent dans l'historique pour /undo.
    """
    scene = get_scene()
    mask  = get_mask()

    if mask is None:
        return {"ok": False, "error": "Faites d'abord un /trace."}
    if not scene:
        return {"ok": False, "error": "Aucune scène chargée."}

    try:
        push_history()

        n_before = scene["n"]
        keep     = ~mask

        for key in ["xyz", "opacity", "scales", "rotations", "colors_sh", "raw"]:
            if key in scene:
                scene[key] = scene[key][keep]

        scene["n"] = int(keep.sum())
        update_scene_fields(**scene)
        clear_mask()

        n_removed = n_before - scene["n"]
        print(f"[Delete] {n_removed:,} gaussiens supprimés, reste {scene['n']:,}")

        out_path = get_out_path("deleted.ply")
        save_scene(out_path)

        return {
            "ok":        True,
            "action":    "delete",
            "ply_path":  out_path,
            "n_removed": n_removed,
        }

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


def action_crop() -> dict:
    """
    Garde uniquement les gaussiens sélectionnés par /trace.
    L'inverse de /delete.
    Sauvegarde l'état précédent dans l'historique pour /undo.
    """
    scene = get_scene()
    mask  = get_mask()

    if mask is None:
        return {"ok": False, "error": "Faites d'abord un /trace."}
    if not scene:
        return {"ok": False, "error": "Aucune scène chargée."}

    try:
        push_history()

        n_before = scene["n"]
        keep     = mask  # garder seulement la sélection

        for key in ["xyz", "opacity", "scales", "rotations", "colors_sh", "raw"]:
            if key in scene:
                scene[key] = scene[key][keep]

        scene["n"] = int(keep.sum())
        update_scene_fields(**scene)
        clear_mask()

        n_kept    = scene["n"]
        n_removed = n_before - n_kept
        print(f"[Crop] {n_kept:,} gaussiens conservés, {n_removed:,} supprimés")

        out_path = get_out_path("cropped.ply")
        save_scene(out_path)

        return {
            "ok":        True,
            "action":    "crop",
            "ply_path":  out_path,
            "n_kept":    n_kept,
            "n_removed": n_removed,
        }

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


def action_undo() -> dict:
    """
    Restaure le dernier état sauvegardé dans l'historique.
    Retourne une erreur si l'historique est vide.
    """
    if history_size() == 0:
        return {
            "ok":    False,
            "error": "Historique vide — aucune action à annuler.",
        }

    try:
        ok = pop_history()
        if not ok:
            return {"ok": False, "error": "Impossible de restaurer l'historique."}

        scene    = get_scene()
        out_path = get_out_path("undo.ply")
        save_scene(out_path)

        print(f"[Undo] Scène restaurée : {scene['n']:,} gaussiens")
        return {
            "ok":              True,
            "action":          "undo",
            "ply_path":        out_path,
            "n":               scene["n"],
            "history_remaining": history_size(),
        }

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}