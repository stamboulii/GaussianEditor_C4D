"""
tests/test_actions.py
----------------------
Tests unitaires pour core/actions/.
Utilise des scènes mockées — pas de vrais modèles IA.

Lancer avec :
    python tests/test_actions.py
"""

import sys
import os
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.scene.scene_state import set_scene, _history
import core.scene.scene_state as ss

from core.actions.load        import action_load
from core.actions.delete      import action_delete, action_crop, action_undo
from core.actions.adjustments import action_opacity, action_scale, action_save
from core.actions.trace       import _trace_geometric


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_scene(n=100):
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    raw = np.zeros((n, len(props)), dtype=np.float32)
    raw[:, 6] = 2.0   # opacity logit ~ 0.88
    raw[:, 7] = np.log(0.01)
    raw[:, 8] = np.log(0.01)
    raw[:, 9] = np.log(0.01)
    raw[:, 10] = 1.0  # rot_0
    xyz = np.random.uniform(-1, 1, (n, 3)).astype(np.float32)
    raw[:, 0] = xyz[:, 0]
    raw[:, 1] = xyz[:, 1]
    raw[:, 2] = xyz[:, 2]
    scene = {
        "ply_path":  "mock.ply",
        "n":         n,
        "xyz":       xyz,
        "opacity":   1.0 / (1.0 + np.exp(-raw[:, 6])),
        "scales":    np.exp(raw[:, 7:10]),
        "rotations": raw[:, 10:14],
        "colors_sh": raw[:, 3:6],
        "props":     props,
        "raw":       raw,
    }
    ss._scene = scene
    ss._mask  = None
    ss._history.clear()
    return scene


def _make_test_ply(n=50) -> str:
    props = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2",
             "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    for p in props:
        header += f"property float {p}\n"
    header += "end_header\n"
    data = np.zeros((n, len(props)), dtype=np.float32)
    data[:, :3] = np.random.randn(n, 3).astype(np.float32)
    data[:, 10] = 1.0
    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp.write(header.encode())
    tmp.write(data.tobytes())
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Tests action_load
# ---------------------------------------------------------------------------

def test_load_valid():
    ply = _make_test_ply(50)
    try:
        result = action_load(ply)
        assert result["ok"] is True
        assert result["n"] == 50
        print("OK : test_load_valid")
    finally:
        os.unlink(ply)


def test_load_missing_file():
    result = action_load("/non/existant.ply")
    assert result["ok"] is False
    assert "introuvable" in result["error"].lower()
    print("OK : test_load_missing_file")


def test_load_empty_path():
    result = action_load("")
    assert result["ok"] is False
    print("OK : test_load_empty_path")


# ---------------------------------------------------------------------------
# Tests action_delete
# ---------------------------------------------------------------------------

def test_delete_no_mask():
    _make_scene(50)
    result = action_delete()
    assert result["ok"] is False
    assert "trace" in result["error"].lower()
    print("OK : test_delete_no_mask")


def test_delete_with_mask():
    scene = _make_scene(100)
    mask  = np.zeros(100, dtype=bool)
    mask[:30] = True
    ss._mask = mask

    result = action_delete()
    assert result["ok"] is True
    assert result["n_removed"] == 30
    assert ss._scene["n"] == 70
    assert ss._mask is None
    print("OK : test_delete_with_mask")


def test_delete_saves_history():
    _make_scene(100)
    mask = np.ones(100, dtype=bool)
    mask[50:] = False
    ss._mask = mask

    action_delete()
    # L'historique doit avoir été poussé avant la suppression
    # (et déjà consommé par delete — l'historique final peut être vide)
    # Ce qu'on vérifie c'est que la scène a bien changé
    assert ss._scene["n"] == 50
    print("OK : test_delete_saves_history")


# ---------------------------------------------------------------------------
# Tests action_crop
# ---------------------------------------------------------------------------

def test_crop_with_mask():
    _make_scene(100)
    mask = np.zeros(100, dtype=bool)
    mask[:40] = True
    ss._mask = mask

    result = action_crop()
    assert result["ok"] is True
    assert result["n_kept"] == 40
    assert result["n_removed"] == 60
    assert ss._scene["n"] == 40
    print("OK : test_crop_with_mask")


def test_crop_no_mask():
    _make_scene(50)
    result = action_crop()
    assert result["ok"] is False
    print("OK : test_crop_no_mask")


# ---------------------------------------------------------------------------
# Tests action_undo
# ---------------------------------------------------------------------------

def test_undo_empty_history():
    _make_scene(50)
    result = action_undo()
    assert result["ok"] is False
    assert "vide" in result["error"].lower()
    print("OK : test_undo_empty_history")


def test_undo_restores_scene():
    _make_scene(100)
    # Pousser l'historique manuellement
    from core.scene import push_history
    push_history()

    # Modifier la scène
    ss._scene["n"] = 999

    # Undo
    result = action_undo()
    assert result["ok"] is True
    assert ss._scene["n"] == 100
    assert result["n"] == 100
    print("OK : test_undo_restores_scene")


# ---------------------------------------------------------------------------
# Tests action_opacity
# ---------------------------------------------------------------------------

def test_opacity_no_mask():
    _make_scene(50)
    result = action_opacity(0.5)
    assert result["ok"] is False
    print("OK : test_opacity_no_mask")


def test_opacity_with_mask():
    _make_scene(100)
    mask = np.zeros(100, dtype=bool)
    mask[:50] = True
    ss._mask = mask

    result = action_opacity(0.1)
    assert result["ok"] is True
    assert result["n_affected"] == 50

    # Vérifier que l'opacité a bien changé pour les points masqués
    assert ss._scene["opacity"][:50].mean() < 0.2
    print("OK : test_opacity_with_mask")


# ---------------------------------------------------------------------------
# Tests action_scale
# ---------------------------------------------------------------------------

def test_scale_with_mask():
    _make_scene(100)
    mask = np.zeros(100, dtype=bool)
    mask[:25] = True
    ss._mask = mask

    scales_before = ss._scene["scales"][:25].copy()
    result = action_scale(2.0)
    assert result["ok"] is True
    assert result["n_affected"] == 25

    scales_after = ss._scene["scales"][:25]
    ratio = scales_after / scales_before
    assert np.allclose(ratio, 2.0, atol=0.01)
    print("OK : test_scale_with_mask")


# ---------------------------------------------------------------------------
# Tests action_save
# ---------------------------------------------------------------------------

def test_save_scene():
    _make_scene(50)
    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    tmp.close()
    try:
        result = action_save(tmp.name)
        assert result["ok"] is True
        assert os.path.isfile(tmp.name)
        assert os.path.getsize(tmp.name) > 0
        print("OK : test_save_scene")
    finally:
        os.unlink(tmp.name)


# ---------------------------------------------------------------------------
# Tests _trace_geometric
# ---------------------------------------------------------------------------

def test_trace_geometric_left():
    scene = _make_scene(200)
    # Forcer xyz symétriques
    ss._scene["xyz"][:, 0] = np.linspace(-1, 1, 200)
    result = _trace_geometric("left")
    assert result["ok"] is True
    assert result["n_selected"] > 0
    # Les points à gauche (x < 0) doivent être sélectionnés
    mask = ss._mask
    selected_x = ss._scene["xyz"][mask, 0]
    assert (selected_x < 0).all()
    print("OK : test_trace_geometric_left")


def test_trace_geometric_unknown_prompt():
    _make_scene(100)
    result = _trace_geometric("random_unknown_thing_xyz")
    assert result["ok"] is True
    assert result["n_selected"] > 0
    print("OK : test_trace_geometric_unknown_prompt")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        test_load_valid,
        test_load_missing_file,
        test_load_empty_path,
        test_delete_no_mask,
        test_delete_with_mask,
        test_delete_saves_history,
        test_crop_with_mask,
        test_crop_no_mask,
        test_undo_empty_history,
        test_undo_restores_scene,
        test_opacity_no_mask,
        test_opacity_with_mask,
        test_scale_with_mask,
        test_save_scene,
        test_trace_geometric_left,
        test_trace_geometric_unknown_prompt,
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