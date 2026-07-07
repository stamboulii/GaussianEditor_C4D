"""
core/scene/scene_ops.py
------------------------
Opérations internes sur la scène : synchronisation raw ↔ attributs,
extension du tableau raw lors de l'ajout de nouveaux gaussiens, fusion.

Extrait de ge_gsplat_server.py → _sync_raw_from_scene(), _rebuild_raw_extended().
"""

import numpy as np
from core.scene.scene_state import get_scene, update_scene_fields


def sync_raw_from_scene() -> None:
    """
    Resynchronise _scene['raw'] depuis les attributs calculés.
    À appeler après toute modification des attributs
    (ex: après action_edit, action_opacity, action_scale).

    Le tableau 'raw' est la source de vérité pour l'export PLY —
    il doit rester cohérent avec xyz, opacity, scales, rotations, colors_sh.
    """
    scene = get_scene()
    props = scene["props"]
    pidx  = {name: i for i, name in enumerate(props)}
    raw   = scene["raw"]

    # XYZ
    for j, k in enumerate(["x", "y", "z"]):
        if k in pidx:
            raw[:, pidx[k]] = scene["xyz"][:, j]

    # Opacité → logit (le PLY stocke en logit-space)
    if "opacity" in pidx:
        op = np.clip(scene["opacity"], 1e-6, 1 - 1e-6)
        raw[:, pidx["opacity"]] = np.log(op / (1 - op))

    # Scales → log (le PLY stocke en log-space)
    for j, k in enumerate(["scale_0", "scale_1", "scale_2"]):
        if k in pidx:
            raw[:, pidx[k]] = np.log(np.clip(scene["scales"][:, j], 1e-8, None))

    # Rotations
    for j, k in enumerate(["rot_0", "rot_1", "rot_2", "rot_3"]):
        if k in pidx:
            raw[:, pidx[k]] = scene["rotations"][:, j]

    # Couleurs SH
    for j, k in enumerate(["f_dc_0", "f_dc_1", "f_dc_2"]):
        if k in pidx:
            raw[:, pidx[k]] = scene["colors_sh"][:, j]

    update_scene_fields(raw=raw)


def rebuild_raw_extended(n_new: int, n_old: int = None) -> None:
    """
    Étend _scene['raw'] pour inclure les nouveaux gaussiens ajoutés.

    À appeler après avoir concaténé les nouveaux gaussiens dans
    scene["xyz"], scene["colors_sh"] etc., mais AVANT de sauvegarder le PLY.

    Args:
        n_new: nombre de nouveaux gaussiens ajoutés
        n_old: taille de raw AVANT l'ajout. Si None, utilise len(scene["raw"])
    """
    scene   = get_scene()
    props   = scene["props"]
    pidx    = {name: i for i, name in enumerate(props)}
    n_total = scene["n"]
    n_props = len(props)

    if n_old is None:
        n_old = len(scene["raw"])

    new_raw = np.zeros((n_total, n_props), dtype=np.float32)
    new_raw[:n_old] = scene["raw"]

    # Extraire les nouveaux attributs
    new_xyz    = scene["xyz"][n_old:]
    new_sh     = scene["colors_sh"][n_old:]
    new_scales = scene["scales"][n_old:] if "scales" in scene else None
    new_op     = scene["opacity"][n_old:] if "opacity" in scene else None

    print(f"[Rebuild] n_old={n_old} n_total={n_total} n_new={n_new}")

    # XYZ
    for j, k in enumerate(["x", "y", "z"]):
        if k in pidx:
            new_raw[n_old:, pidx[k]] = new_xyz[:, j]

    # Opacité → logit
    if "opacity" in pidx:
        if new_op is not None and len(new_op) == n_new:
            op = np.clip(new_op, 1e-6, 1 - 1e-6)
            new_raw[n_old:, pidx["opacity"]] = np.log(op / (1 - op))
        else:
            new_raw[n_old:, pidx["opacity"]] = 1.735  # logit(0.85)

    # Couleurs SH
    for j, k in enumerate(["f_dc_0", "f_dc_1", "f_dc_2"]):
        if k in pidx:
            new_raw[n_old:, pidx[k]] = new_sh[:, j]

    # Scales → log
    for j, k in enumerate(["scale_0", "scale_1", "scale_2"]):
        if k in pidx:
            if new_scales is not None and len(new_scales) == n_new:
                log_scales = np.log(np.clip(new_scales[:, j], 1e-8, None))
                new_raw[n_old:, pidx[k]] = log_scales
            else:
                scale_val = max(
                    np.median(np.exp(scene["raw"][:n_old, pidx[k]])), 0.001
                )
                new_raw[n_old:, pidx[k]] = np.log(scale_val)

    # Rotations — quaternion identité par défaut
    if "rot_0" in pidx:
        new_raw[n_old:, pidx["rot_0"]] = 1.0

    update_scene_fields(raw=new_raw)


def merge_gaussians(new_gaussians: dict) -> int:
    """
    Fusionne un nouveau dict de gaussiens dans la scène courante.
    Retourne le nombre de gaussiens ajoutés.

    Args:
        new_gaussians: dict avec clés xyz, colors_sh, opacity, scales, rotations, n

    Returns:
        n_added: nombre de gaussiens effectivement ajoutés
    """
    scene   = get_scene()
    n_added = new_gaussians.get("n", 0)
    if n_added == 0:
        return 0

    n_old = len(scene["raw"])

    for key in ["xyz", "opacity", "scales", "rotations", "colors_sh"]:
        if key in scene and key in new_gaussians:
            scene[key] = np.concatenate([scene[key], new_gaussians[key]], axis=0)

    scene["n"] = len(scene["xyz"])
    update_scene_fields(**scene)

    rebuild_raw_extended(n_added, n_old=n_old)
    return n_added