"""
core/actions/__init__.py
-------------------------
Exports publics du module actions.

Chaque action correspond à un endpoint HTTP du serveur.
"""

from core.actions.generate    import action_generate, action_generate_and_add
from core.actions.load        import action_load
from core.actions.delete      import action_delete, action_crop, action_undo
from core.actions.adjustments import action_opacity, action_scale, action_save, action_export_splat
from core.actions.trace       import action_trace

__all__ = [
    "action_generate", "action_generate_and_add",
    "action_load",
    "action_delete", "action_crop", "action_undo",
    "action_opacity", "action_scale", "action_save", "action_export_splat",
    "action_trace",
]