"""
core/utils/ortho_render.py
---------------------------
Rendu orthographique de nuages de gaussiens et agrégation de masques 3D.

Extrait de ge_gsplat_server.py → _render_ortho_views(), _aggregate_masks_3d().
Utilisé par : core/actions/trace.py, core/actions/edit.py
"""

import numpy as np
from typing import List, Tuple


def render_ortho_views(xyz: np.ndarray, colors: np.ndarray, size: int = 512) -> list:
    """
    Rendu orthographique sur 3 axes (front, side, top).

    Args:
        xyz: positions (N, 3) float
        colors: couleurs RGB (N, 3) float dans [0,1]
        size: résolution de l'image de rendu

    Returns:
        Liste de (PIL.Image, cam_info) pour chaque vue
    """
    from PIL import Image

    views = []
    view_configs = [
        (2, [0, 1], "front"),
        (0, [1, 2], "side"),
        (1, [0, 2], "top"),
    ]

    for _, axes, name in view_configs:
        pts  = xyz[:, axes]
        mn   = pts.min(0)
        mx   = pts.max(0)
        span = mx - mn
        span[span < 1e-8] = 1.0

        img_arr = np.zeros((size, size, 3), dtype=np.uint8)
        pxs = np.clip(((pts[:, 0] - mn[0]) / span[0] * (size - 1)).astype(int), 0, size - 1)
        pys = np.clip(((pts[:, 1] - mn[1]) / span[1] * (size - 1)).astype(int), 0, size - 1)
        img_arr[pys, pxs] = (np.clip(colors, 0, 1) * 255).astype(np.uint8)

        cam_info = {
            "type": "ortho",
            "name": name,
            "axes": axes,
            "mn":   mn,
            "mx":   mx,
            "size": size,
            "xyz":  xyz,
        }
        views.append((Image.fromarray(img_arr), cam_info))

    return views


def aggregate_masks_3d(
    xyz: np.ndarray,
    views_masks: list,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Agrège les masques 2D de plusieurs vues en un masque 3D par vote majoritaire.

    Args:
        xyz: positions des gaussiens (N, 3)
        views_masks: liste de (mask_2d, cam_info) — masques booléens 2D
        threshold: fraction de vues minimum pour sélectionner un gaussien

    Returns:
        np.ndarray bool (N,) — mask 3D des gaussiens sélectionnés
    """
    n    = len(xyz)
    vote = np.zeros(n, dtype=np.float32)

    for mask_2d, cam_info in views_masks:
        axes = cam_info["axes"]
        mn   = cam_info["mn"]
        mx   = cam_info["mx"]
        size = cam_info["size"]
        span = mx - mn
        span[span < 1e-8] = 1.0

        pts = xyz[:, axes]
        pxs = np.clip(((pts[:, 0] - mn[0]) / span[0] * (size - 1)).astype(int), 0, size - 1)
        pys = np.clip(((pts[:, 1] - mn[1]) / span[1] * (size - 1)).astype(int), 0, size - 1)
        vote += mask_2d[pys, pxs].astype(np.float32)

    vote /= max(len(views_masks), 1)
    return vote >= threshold