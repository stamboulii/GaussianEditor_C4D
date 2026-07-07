"""
tests/test_scene.py
--------------------
Tests unitaires pour core/scene/.

Lancer avec :
    python tests/test_scene.py
"""

import sys
import os
import tempfile
import struct
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.scene import (
    load_scene, save_scene, get_scene, get_mask,
    set_mask, clear_mask, is_scene_loaded,
    push_history, pop_history, history_size,
    sync_raw_from_scene, merge_gaussians,
)
from core.scene.scene_state import set_scene


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_ply(n=100) -> str:
    """Crée un PLY Gaussian Splatting minimal en mémoire, retourne le chemin."""
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    for p in props:
        header += f"property float {p}\n"
    header += "end_header\n"

    data = np.zeros((n, len(props)), dtype=np.float32)
    data[:, 0] = np.random.uniform(-1, 1, n)   # x
    data[:, 1] = np.random.uniform(-1, 1, n)   # y
    data[:, 2] = np.random.uniform(-1, 1, n)   # z
    data[:, 3] = np.random.uniform(-0.5, 0.5, n)  # f_dc_0
    data[:, 4] = np.random.uniform(-0.5, 0.5, n)  # f_dc_1
    data[:, 5] = np.random.uniform(-0.5, 0.5, n)  # f_dc_2
    data[:, 6] = np.random.uniform(-2, 2, n)   # opacity (logit-space)
    data[:, 7:10] = np.log(np.random.uniform(0.001, 0.05, (n, 3)))  # scales log
    data[:, 10] = 1.0  # rot_0 = 1 (quaternion identité)

    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp.write(header.encode("utf-8"))
    tmp.write(data.tobytes())
    tmp.close()
    return tmp.name


def _make_minimal_scene(n=50) -> dict:
    """Crée un dict scene minimal pour les tests sans fichier PLY."""
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    raw = np.zeros((n, len(props)), dtype=np.float32)
    raw[:, 10] = 1.0  # rot_0
    return {
        "ply_path":  "test.ply",
        "n":         n,
        "xyz":       raw[:, :3],
        "opacity":   np.full(n, 0.9, dtype=np.float32),
        "scales":    np.full((n, 3), 0.01, dtype=np.float32),
        "rotations": np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
        "colors_sh": np.zeros((n, 3), dtype=np.float32),
        "props":     props,
        "raw":       raw,
    }


# ---------------------------------------------------------------------------
# Tests load_scene
# ---------------------------------------------------------------------------

def test_load_scene_basic():
    """Charger un PLY valide doit remplir la scène correctement."""
    ply = _make_test_ply(100)
    try:
        ok = load_scene(ply)
        assert ok, "load_scene doit retourner True"
        scene = get_scene()
        assert scene["n"] == 100
        assert scene["xyz"].shape == (100, 3)
        assert scene["colors_sh"].shape == (100, 3)
        assert scene["opacity"].shape == (100,)
        assert scene["scales"].shape == (100, 3)
        assert "raw" in scene
        print("OK : test_load_scene_basic")
    finally:
        os.unlink(ply)


def test_load_scene_opacity_conversion():
    """L'opacité doit être convertie depuis logit-space vers [0,1]."""
    ply = _make_test_ply(50)
    try:
        load_scene(ply)
        scene = get_scene()
        assert scene["opacity"].min() >= 0.0
        assert scene["opacity"].max() <= 1.0
        print("OK : test_load_scene_opacity_conversion")
    finally:
        os.unlink(ply)


def test_load_scene_scales_positive():
    """Les scales doivent être positifs (exp du log-space PLY)."""
    ply = _make_test_ply(50)
    try:
        load_scene(ply)
        scene = get_scene()
        assert (scene["scales"] > 0).all()
        print("OK : test_load_scene_scales_positive")
    finally:
        os.unlink(ply)


def test_load_scene_invalid_path():
    """Un chemin invalide doit retourner False sans crasher."""
    ok = load_scene("/chemin/qui/nexiste/pas.ply")
    assert ok is False
    print("OK : test_load_scene_invalid_path")


# ---------------------------------------------------------------------------
# Tests save_scene
# ---------------------------------------------------------------------------

def test_save_scene_roundtrip():
    """save_scene puis load_scene doit redonner la même scène."""
    ply_in = _make_test_ply(80)
    try:
        load_scene(ply_in)
        scene_before = get_scene()
        n_before = scene_before["n"]

        tmp_out = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
        tmp_out.close()

        ok = save_scene(tmp_out.name)
        assert ok, "save_scene doit retourner True"

        ok2 = load_scene(tmp_out.name)
        assert ok2
        scene_after = get_scene()
        assert scene_after["n"] == n_before
        assert np.allclose(scene_after["xyz"], scene_before["xyz"], atol=1e-5)
        print("OK : test_save_scene_roundtrip")
    finally:
        os.unlink(ply_in)
        os.unlink(tmp_out.name)


def test_save_scene_empty():
    """save_scene sur une scène vide doit retourner False."""
    from core.scene.scene_state import _scene
    import core.scene.scene_state as ss
    original = ss._scene.copy()
    ss._scene = {}
    ok = save_scene("/tmp/should_not_exist.ply")
    assert ok is False
    ss._scene = original
    print("OK : test_save_scene_empty")


# ---------------------------------------------------------------------------
# Tests mask
# ---------------------------------------------------------------------------

def test_mask_set_clear():
    """set_mask et clear_mask doivent fonctionner correctement."""
    mask = np.array([True, False, True])
    set_mask(mask)
    assert (get_mask() == mask).all()
    clear_mask()
    assert get_mask() is None
    print("OK : test_mask_set_clear")


# ---------------------------------------------------------------------------
# Tests historique (undo)
# ---------------------------------------------------------------------------

def test_push_pop_history():
    """push_history puis pop_history doit restaurer la scène."""
    scene = _make_minimal_scene(30)
    set_scene(scene)

    original_n = get_scene()["n"]
    push_history()
    assert history_size() == 1

    # Modifier la scène
    get_scene()["n"] = 999

    # Undo
    ok = pop_history()
    assert ok
    assert get_scene()["n"] == original_n
    assert history_size() == 0
    print("OK : test_push_pop_history")


def test_pop_history_empty():
    """pop_history sur historique vide doit retourner False."""
    import core.scene.scene_state as ss
    ss._history = []
    ok = pop_history()
    assert ok is False
    print("OK : test_pop_history_empty")


# ---------------------------------------------------------------------------
# Tests merge_gaussians
# ---------------------------------------------------------------------------

def test_merge_gaussians():
    """merge_gaussians doit augmenter le nombre de gaussiens."""
    scene = _make_minimal_scene(50)
    set_scene(scene)

    n_before = get_scene()["n"]
    new_g = {
        "n":         20,
        "xyz":       np.random.randn(20, 3).astype(np.float32),
        "colors_sh": np.zeros((20, 3), dtype=np.float32),
        "opacity":   np.full(20, 0.9, dtype=np.float32),
        "scales":    np.full((20, 3), 0.01, dtype=np.float32),
        "rotations": np.tile([1, 0, 0, 0], (20, 1)).astype(np.float32),
    }

    n_added = merge_gaussians(new_g)
    assert n_added == 20
    assert get_scene()["n"] == n_before + 20
    assert get_scene()["xyz"].shape == (n_before + 20, 3)
    print("OK : test_merge_gaussians")


def test_merge_gaussians_empty():
    """merge_gaussians avec n=0 ne doit rien changer."""
    scene = _make_minimal_scene(50)
    set_scene(scene)
    n_before = get_scene()["n"]

    n_added = merge_gaussians({"n": 0})
    assert n_added == 0
    assert get_scene()["n"] == n_before
    print("OK : test_merge_gaussians_empty")


# ---------------------------------------------------------------------------
# Tests sync_raw
# ---------------------------------------------------------------------------

def test_sync_raw_from_scene():
    """sync_raw_from_scene doit mettre à jour raw depuis les attributs."""
    scene = _make_minimal_scene(20)
    set_scene(scene)

    # Modifier xyz directement
    get_scene()["xyz"][0] = [99.0, 88.0, 77.0]
    sync_raw_from_scene()

    props = get_scene()["props"]
    pidx  = {name: i for i, name in enumerate(props)}
    raw   = get_scene()["raw"]

    assert abs(raw[0, pidx["x"]] - 99.0) < 1e-4
    assert abs(raw[0, pidx["y"]] - 88.0) < 1e-4
    assert abs(raw[0, pidx["z"]] - 77.0) < 1e-4
    print("OK : test_sync_raw_from_scene")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        test_load_scene_basic,
        test_load_scene_opacity_conversion,
        test_load_scene_scales_positive,
        test_load_scene_invalid_path,
        test_save_scene_roundtrip,
        test_save_scene_empty,
        test_mask_set_clear,
        test_push_pop_history,
        test_pop_history_empty,
        test_merge_gaussians,
        test_merge_gaussians_empty,
        test_sync_raw_from_scene,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"ECHEC : {t.__name__} → {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"ERREUR : {t.__name__} → {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{len(tests) - failed}/{len(tests)} tests réussis")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()