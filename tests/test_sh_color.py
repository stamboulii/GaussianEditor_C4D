"""
tests/test_sh_color.py
-----------------------
Tests unitaires pour la conversion SH <-> RGB.

Lancer avec :
    python -m pytest tests/test_sh_color.py -v
ou directement :
    python tests/test_sh_color.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.utils.sh_color import sh_to_rgb, rgb_to_sh, resolve_colors_rgb, average_color, SH_C0


def test_sh_to_rgb_zero():
    """SH = 0 doit donner RGB = 0.5 (gris neutre)."""
    sh = np.array([[0.0, 0.0, 0.0]])
    rgb = sh_to_rgb(sh)
    assert np.allclose(rgb, [[0.5, 0.5, 0.5]]), f"Attendu [0.5,0.5,0.5], obtenu {rgb}"
    print("OK : test_sh_to_rgb_zero")


def test_sh_to_rgb_clipping():
    """Des valeurs SH extrêmes doivent être clampées dans [0, 1]."""
    sh = np.array([[100.0, -100.0, 0.0]])
    rgb = sh_to_rgb(sh)
    assert rgb[0, 0] == 1.0, "Valeur haute doit clamper a 1.0"
    assert rgb[0, 1] == 0.0, "Valeur basse doit clamper a 0.0"
    print("OK : test_sh_to_rgb_clipping")


def test_roundtrip_sh_rgb():
    """sh_to_rgb puis rgb_to_sh doit redonner la valeur initiale (si pas clampee)."""
    original_sh = np.array([[0.3, -0.2, 0.1]])
    rgb = sh_to_rgb(original_sh)
    back_to_sh = rgb_to_sh(rgb)
    assert np.allclose(back_to_sh, original_sh, atol=1e-5), \
        f"Roundtrip echoue : {original_sh} -> {rgb} -> {back_to_sh}"
    print("OK : test_roundtrip_sh_rgb")


def test_real_world_example():
    """
    Cas reel observe pendant le debug : couleur moyenne du piano.
    SH brut donnant un gris fonce coherent avec un piano noir.
    """
    sh = np.array([[-1.024, -1.060, -1.084]])  # valeurs negatives typiques
    rgb = sh_to_rgb(sh)
    # Doit donner une couleur sombre (proche du noir), pas du orange
    assert rgb[0, 0] < 0.3 and rgb[0, 1] < 0.3 and rgb[0, 2] < 0.3, \
        f"Couleur attendue sombre, obtenu {rgb}"
    print(f"OK : test_real_world_example -> RGB = {rgb[0]}")


def test_resolve_colors_rgb_priority_sh():
    """resolve_colors_rgb doit privilegier colors_sh si present."""
    sh  = np.array([[0.0, 0.0, 0.0]])
    rgb_alt = np.array([[0.9, 0.9, 0.9]])
    result = resolve_colors_rgb(colors_sh=sh, colors_rgb=rgb_alt)
    assert np.allclose(result, [[0.5, 0.5, 0.5]]), "Doit utiliser colors_sh, pas colors_rgb"
    print("OK : test_resolve_colors_rgb_priority_sh")


def test_resolve_colors_rgb_fallback_rgb():
    """resolve_colors_rgb doit utiliser colors_rgb si colors_sh absent."""
    result = resolve_colors_rgb(colors_sh=None, colors_rgb=[[0.7, 0.2, 0.1]])
    assert np.allclose(result, [[0.7, 0.2, 0.1]])
    print("OK : test_resolve_colors_rgb_fallback_rgb")


def test_resolve_colors_rgb_fallback_gray():
    """resolve_colors_rgb doit retourner du gris si aucune couleur n'est disponible."""
    result = resolve_colors_rgb(colors_sh=None, colors_rgb=None, n_fallback=3)
    assert result.shape == (3, 3)
    assert np.allclose(result, 0.5)
    print("OK : test_resolve_colors_rgb_fallback_gray")


def test_average_color():
    """average_color doit calculer une moyenne correcte."""
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    avg = average_color(colors)
    expected = (1/3, 1/3, 1/3)
    assert np.allclose(avg, expected, atol=1e-5), f"Attendu {expected}, obtenu {avg}"
    print("OK : test_average_color")


def test_average_color_empty():
    """average_color sur un tableau vide doit retourner du gris sans crasher."""
    avg = average_color(np.array([]).reshape(0, 3))
    assert avg == (0.5, 0.5, 0.5)
    print("OK : test_average_color_empty")


def test_sh_c0_constant():
    """Verifie que la constante SH_C0 est la bonne valeur standard 3DGS."""
    assert abs(SH_C0 - 0.282095) < 1e-6
    print("OK : test_sh_c0_constant")


def run_all_tests():
    tests = [
        test_sh_to_rgb_zero,
        test_sh_to_rgb_clipping,
        test_roundtrip_sh_rgb,
        test_real_world_example,
        test_resolve_colors_rgb_priority_sh,
        test_resolve_colors_rgb_fallback_rgb,
        test_resolve_colors_rgb_fallback_gray,
        test_average_color,
        test_average_color_empty,
        test_sh_c0_constant,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"ECHEC : {t.__name__} -> {e}")
            failed += 1
        except Exception as e:
            print(f"ERREUR : {t.__name__} -> {e}")
            failed += 1

    print(f"\n{len(tests) - failed}/{len(tests)} tests reussis")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()