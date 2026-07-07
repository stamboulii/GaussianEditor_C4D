"""
core/scene/__init__.py
-----------------------
Exports publics du module scene.

Usage depuis le reste du projet :
    from core.scene import load_scene, save_scene, get_scene, get_mask, ...
"""

from core.scene.scene_state import (
    get_scene, get_mask, set_mask, clear_mask, is_scene_loaded,
    push_history, pop_history, history_size, set_scene, update_scene_fields,
)
from core.scene.ply_loader import load_scene
from core.scene.ply_writer import save_scene
from core.scene.scene_ops  import (
    sync_raw_from_scene, rebuild_raw_extended, merge_gaussians,
)

__all__ = [
    "get_scene", "get_mask", "set_mask", "clear_mask", "is_scene_loaded",
    "push_history", "pop_history", "history_size",
    "set_scene", "update_scene_fields",
    "load_scene", "save_scene",
    "sync_raw_from_scene", "rebuild_raw_extended", "merge_gaussians",
]