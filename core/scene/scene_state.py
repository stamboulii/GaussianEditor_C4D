"""
core/scene/scene_state.py
--------------------------
État global de la scène Gaussian Splatting.

Contient le dict _scene, le masque _mask, et l'historique _history pour /undo.
Toutes les actions lisent/écrivent via les fonctions de ce module —
jamais en accédant directement aux variables globales depuis l'extérieur.
"""

import threading
import numpy as np

# ---------------------------------------------------------------------------
# État global
# ---------------------------------------------------------------------------

_scene: dict   = {}    # xyz, colors_sh, opacity, scales, rotations, raw, props, n, ply_path
_mask          = None  # numpy bool array — gaussiens sélectionnés par /trace
_history: list = []    # pile d'états pour /undo (max 5)
_lock          = threading.Lock()

_HISTORY_MAX = 5


# ---------------------------------------------------------------------------
# Accesseurs
# ---------------------------------------------------------------------------

def get_scene() -> dict:
    return _scene


def get_mask():
    return _mask


def set_mask(mask) -> None:
    global _mask
    _mask = mask


def clear_mask() -> None:
    global _mask
    _mask = None


def is_scene_loaded() -> bool:
    return bool(_scene) and "xyz" in _scene


# ---------------------------------------------------------------------------
# Historique (undo)
# ---------------------------------------------------------------------------

def push_history() -> None:
    """
    Sauvegarde un snapshot de la scène courante dans la pile d'historique.
    À appeler AVANT toute action destructive (delete, edit, opacity, scale).
    Limite la pile à _HISTORY_MAX états pour ne pas exploser la RAM.
    """
    global _history
    if not _scene:
        return
    snapshot = {
        k: v.copy() if isinstance(v, np.ndarray) else v
        for k, v in _scene.items()
        if k != "raw"  # raw est reconstruit depuis les autres — pas besoin de le dupliquer
    }
    # Sauvegarder raw séparément (np.ndarray)
    if "raw" in _scene:
        snapshot["raw"] = _scene["raw"].copy()

    _history.append(snapshot)
    if len(_history) > _HISTORY_MAX:
        _history.pop(0)


def pop_history() -> bool:
    """
    Restaure le dernier état sauvegardé.
    Retourne True si la restauration a réussi, False si l'historique est vide.
    """
    global _scene, _mask
    if not _history:
        return False
    _scene = _history.pop()
    _mask  = None  # reset du masque après undo
    return True


def history_size() -> int:
    return len(_history)


# ---------------------------------------------------------------------------
# Mise à jour de la scène
# ---------------------------------------------------------------------------

def set_scene(new_scene: dict) -> None:
    """Remplace complètement la scène courante."""
    global _scene, _mask
    _scene = new_scene
    _mask  = None


def update_scene_fields(**kwargs) -> None:
    """Met à jour des champs spécifiques de la scène sans tout remplacer."""
    global _scene
    _scene.update(kwargs)