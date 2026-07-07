"""
core/utils/sh_color.py
-----------------------
Conversion entre Spherical Harmonics (SH) et couleurs RGB.

Les fichiers PLY Gaussian Splatting stockent la couleur sous forme de
coefficient DC (degré 0) des Spherical Harmonics, pas en RGB direct.
Cette conversion est nécessaire partout où on affiche ou exporte une couleur :
  - viewer C4D (affichage)
  - export PLY / splat
  - rendu orthographique (trace, edit)

Formule standard 3D Gaussian Splatting (Kerbl et al. 2023) :
    RGB = 0.5 + SH_C0 * f_dc
    f_dc = (RGB - 0.5) / SH_C0

SH_C0 = 0.282095 (constante d'harmonique sphérique de degré 0)
"""

import numpy as np

# Constante d'harmonique sphérique de degré 0 (terme DC)
SH_C0 = 0.282095


def sh_to_rgb(colors_sh) -> np.ndarray:
    """
    Convertit des couleurs SH (degré 0) en RGB normalisé [0, 1].

    Args:
        colors_sh: array-like de forme (N, 3), valeurs SH brutes
                   (peuvent être négatives, hors range [0,1])

    Returns:
        np.ndarray (N, 3) float32, valeurs RGB clampées dans [0, 1]
    """
    arr = np.asarray(colors_sh, dtype=np.float32)
    rgb = 0.5 + SH_C0 * arr
    return np.clip(rgb, 0.0, 1.0)


def rgb_to_sh(colors_rgb) -> np.ndarray:
    """
    Convertit des couleurs RGB [0, 1] en SH (degré 0) brut.

    Args:
        colors_rgb: array-like de forme (N, 3), valeurs RGB dans [0, 1]

    Returns:
        np.ndarray (N, 3) float32, valeurs SH (peuvent sortir de [0,1] inversé)
    """
    arr = np.asarray(colors_rgb, dtype=np.float32)
    return (arr - 0.5) / SH_C0


def resolve_colors_rgb(colors_sh=None, colors_rgb=None, n_fallback=1, fallback_gray=0.5):
    """
    Résout la meilleure source de couleur disponible vers du RGB [0,1].
    Priorité : colors_sh > colors_rgb > gris neutre.

    C'est la fonction à utiliser dans le viewer et les actions —
    elle centralise la logique de fallback qui était dupliquée
    dans plusieurs fichiers (gsplat_viewer.py, ge_gsplat_server.py).

    Args:
        colors_sh: couleurs SH brutes ou None
        colors_rgb: couleurs RGB déjà calculées ou None
        n_fallback: nombre de points si aucune couleur disponible
        fallback_gray: valeur de gris par défaut [0,1]

    Returns:
        np.ndarray (N, 3) float32, RGB dans [0, 1]
    """
    if _has_data(colors_sh):
        return sh_to_rgb(colors_sh)
    if _has_data(colors_rgb):
        arr = np.asarray(colors_rgb, dtype=np.float32)
        return np.clip(arr, 0.0, 1.0)
    return np.full((n_fallback, 3), fallback_gray, dtype=np.float32)


def average_color(colors_rgb, sample_step=None, max_samples=2000) -> tuple:
    """
    Calcule la couleur moyenne d'un ensemble de couleurs RGB,
    avec sous-échantillonnage pour rester performant sur de gros nuages.

    Args:
        colors_rgb: np.ndarray (N, 3) RGB dans [0, 1]
        sample_step: pas d'échantillonnage explicite, ou None pour auto
        max_samples: nombre max d'échantillons si sample_step est None

    Returns:
        tuple (r, g, b) float
    """
    arr = np.asarray(colors_rgb, dtype=np.float32)
    n = len(arr)
    if n == 0:
        return (0.5, 0.5, 0.5)

    step = sample_step or max(1, n // max_samples)
    sample = arr[::step]
    avg = sample.mean(axis=0)
    return (float(avg[0]), float(avg[1]), float(avg[2]))


def _has_data(arr) -> bool:
    """Vérifie si un tableau (list ou numpy array) est non-vide, sans lever d'exception."""
    if arr is None:
        return False
    try:
        return len(arr) > 0
    except Exception:
        return False