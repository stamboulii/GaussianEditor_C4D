"""
core/depth_estimator.py
------------------------
Estimation de profondeur monoculaire pour GaussianEditor C4D.

Utilise Depth Anything V2 Small (~100MB) de HuggingFace.
Compatible RTX 2050 (4GB VRAM) — modèle très léger.

Rôle dans le pipeline Add :
  image 2D inpaintée
       ↓
  DepthEstimator.estimate(image)
       ↓
  carte de profondeur normalisée [0, 1]
       ↓
  wonder3d_client.py → lift 2D→3D

Références :
  - Depth Anything V2 : https://depth-anything-v2.github.io
  - Yang et al. NIPS 2024
"""

import os
import traceback
from typing import Optional, Union
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Détection hardware
# ---------------------------------------------------------------------------

def _detect_device():
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu", 0.0
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return "cuda", vram
    except Exception:
        return "cpu", 0.0


DEVICE, VRAM_GB = _detect_device()


# ---------------------------------------------------------------------------
# Modèles disponibles par taille
# ---------------------------------------------------------------------------

DEPTH_MODELS = {
    "small":  "depth-anything/Depth-Anything-V2-Small-hf",   # ~100MB ✅ RTX 2050
    "base":   "depth-anything/Depth-Anything-V2-Base-hf",    # ~400MB ⚠️
    "large":  "depth-anything/Depth-Anything-V2-Large-hf",   # ~1.3GB ❌ RTX 2050
}

# Sélection automatique selon VRAM
def _auto_select_model():
    if VRAM_GB >= 6:
        return "base"
    return "small"


# ---------------------------------------------------------------------------
# DepthEstimator
# ---------------------------------------------------------------------------

class DepthEstimator:
    """
    Estimateur de profondeur monoculaire basé sur Depth Anything V2.

    Utilisation :
        estimator = DepthEstimator()
        depth_map = estimator.estimate(pil_image)
        # depth_map : numpy (H, W) float32, normalisé [0, 1]
        # 0 = proche, 1 = loin (convention depth-anything)

    Lazy loading : le modèle est chargé au premier appel estimate().
    """

    def __init__(self, model_size: str = "auto"):
        """
        Args:
            model_size : "small" | "base" | "large" | "auto"
                         "auto" sélectionne selon la VRAM disponible.
        """
        if model_size == "auto":
            model_size = _auto_select_model()

        if model_size not in DEPTH_MODELS:
            raise ValueError(
                f"model_size doit être : {list(DEPTH_MODELS.keys())} | 'auto'"
            )

        self.model_size = model_size
        self.model_id   = DEPTH_MODELS[model_size]
        self._pipe      = None

        print(f"[Depth] Config : {model_size} ({self.model_id})")
        print(f"[Depth] Device : {DEVICE} | VRAM : {VRAM_GB:.1f}GB")

    def _load(self):
        """Charge le pipeline HuggingFace (lazy)."""
        if self._pipe is not None:
            return

        from transformers import pipeline as hf_pipeline

        print(f"[Depth] Chargement {self.model_id}...")
        device_idx = 0 if DEVICE == "cuda" else -1

        self._pipe = hf_pipeline(
            "depth-estimation",
            model=self.model_id,
            device=device_idx,
        )
        print(f"[Depth] Modèle prêt sur {DEVICE}")

    def estimate(self, image) -> np.ndarray:
        """
        Estime la carte de profondeur d'une image.

        Args:
            image : PIL.Image.Image ou numpy (H, W, 3) uint8

        Returns:
            depth_map : numpy (H, W) float32
                        Valeurs normalisées [0, 1]
                        0 = proche caméra, 1 = loin
        """
        from PIL import Image

        self._load()

        # Convertir en PIL si nécessaire
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(image)
        else:
            pil_img = image

        # Taille optimale pour Depth Anything V2
        # Le modèle accepte toute taille mais 518x518 est optimal
        orig_size = pil_img.size
        pil_resized = pil_img.resize((518, 518))

        result    = self._pipe(pil_resized)
        depth_raw = np.array(result["depth"], dtype=np.float32)

        # Normaliser [0, 1]
        d_min = depth_raw.min()
        d_max = depth_raw.max()
        if d_max - d_min < 1e-8:
            depth_norm = np.zeros_like(depth_raw)
        else:
            depth_norm = (depth_raw - d_min) / (d_max - d_min)

        # Redimensionner à la taille originale
        depth_pil    = Image.fromarray((depth_norm * 65535).astype(np.uint16))
        depth_resized = depth_pil.resize(orig_size, Image.BILINEAR)
        depth_final  = np.array(depth_resized, dtype=np.float32) / 65535.0

        return depth_final

    def estimate_world_z(self, image, xyz_ref: np.ndarray,
                         mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Estime les coordonnées Z monde pour les pixels d'un masque.

        Mappe la profondeur normalisée [0,1] vers la plage Z
        de la scène de référence xyz_ref.

        Args:
            image    : image PIL ou numpy
            xyz_ref  : (N, 3) — gaussiens existants (pour calibrer Z)
            mask     : (H, W) bool — pixels d'intérêt.
                       Si None, retourne la carte complète.

        Returns:
            Si mask fourni :
                z_values : (M,) float32 — Z monde pour chaque pixel masqué
            Sinon :
                depth_world : (H, W) float32 — carte Z monde complète
        """
        depth_norm = self.estimate(image)

        # Calibrer vers les Z de la scène
        z_min = float(xyz_ref[:, 2].min())
        z_max = float(xyz_ref[:, 2].max())
        z_range = max(z_max - z_min, 0.01)

        depth_world = z_min + depth_norm * z_range

        if mask is not None:
            ys, xs = np.where(mask)
            return depth_world[ys, xs]
        return depth_world

    def unload(self):
        """Libère la mémoire GPU."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            try:
                import torch
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                print("[Depth] Modèle déchargé")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Fallback sans modèle — interpolation depuis les voisins
# ---------------------------------------------------------------------------

def interpolate_z_from_neighbors(query_xy: np.ndarray,
                                  xyz_ref: np.ndarray,
                                  k: int = 5) -> np.ndarray:
    """
    Estime Z par inverse distance weighting depuis les gaussiens voisins.

    Utilisé quand Depth Anything n'est pas disponible.

    Args:
        query_xy : (M, 2) — coordonnées XY monde des nouveaux points
        xyz_ref  : (N, 3) — gaussiens existants
        k        : nombre de voisins

    Returns:
        z_values : (M,) float32
    """
    try:
        from scipy.spatial import KDTree
        tree   = KDTree(xyz_ref[:, :2])
        k_real = min(k, len(xyz_ref))
        dists, idxs = tree.query(query_xy, k=k_real)

        # IDW
        if k_real == 1:
            return xyz_ref[idxs, 2].astype(np.float32)

        weights = 1.0 / (dists + 1e-8)
        weights /= weights.sum(axis=1, keepdims=True)
        return (xyz_ref[idxs, 2] * weights).sum(axis=1).astype(np.float32)

    except ImportError:
        # Pas de scipy → moyenne globale
        print("[Depth] scipy absent, Z = moyenne globale")
        return np.full(len(query_xy), xyz_ref[:, 2].mean(), dtype=np.float32)


# ---------------------------------------------------------------------------
# Fonction utilitaire — utilisée depuis wonder3d_client.py
# ---------------------------------------------------------------------------

def estimate_depth(image,
                   xyz_ref: Optional[np.ndarray] = None,
                   mask: Optional[np.ndarray] = None,
                   model_size: str = "auto") -> np.ndarray:
    """
    Point d'entrée simplifié.

    Args:
        image      : PIL.Image ou numpy (H, W, 3)
        xyz_ref    : (N, 3) gaussiens existants (pour calibration Z)
                     Si fourni, retourne des Z monde
        mask       : (H, W) bool — si fourni avec xyz_ref,
                     retourne Z monde pour les pixels masqués seulement
        model_size : "small" | "base" | "large" | "auto"

    Returns:
        numpy float32 :
          - (H, W) depth normalisée [0,1] si xyz_ref=None
          - (H, W) Z monde si xyz_ref fourni et mask=None
          - (M,) Z monde si xyz_ref et mask fournis
    """
    try:
        estimator = DepthEstimator(model_size=model_size)
        if xyz_ref is not None:
            return estimator.estimate_world_z(image, xyz_ref, mask)
        return estimator.estimate(image)

    except Exception as e:
        print(f"[Depth] Erreur ({e}), fallback interpolation")
        if xyz_ref is not None and mask is not None:
            from PIL import Image as PILImage
            if isinstance(image, np.ndarray):
                h, w = image.shape[:2]
            else:
                w, h = image.size
            ys, xs = np.where(mask)
            # Coordonnées XY monde approximatives
            query_xy = np.stack([
                xs / w * (xyz_ref[:, 0].max() - xyz_ref[:, 0].min()) + xyz_ref[:, 0].min(),
                ys / h * (xyz_ref[:, 1].max() - xyz_ref[:, 1].min()) + xyz_ref[:, 1].min(),
            ], axis=1)
            return interpolate_z_from_neighbors(query_xy, xyz_ref)
        # Retourner une carte plate
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = image.size
        return np.full((h, w), 0.5, dtype=np.float32)


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Test : estimation de profondeur sur une image synthétique.

    Usage :
        cd GaussianEditor_C4D
        python core/depth_estimator.py
        python core/depth_estimator.py --image path/to/image.png
    """
    import sys
    from PIL import Image

    print("=== Test DepthEstimator ===")
    print(f"Device : {DEVICE} | VRAM : {VRAM_GB:.1f}GB")

    # Image de test
    if len(sys.argv) > 2 and sys.argv[1] == "--image":
        img_path = sys.argv[2]
        if not os.path.isfile(img_path):
            print(f"Image introuvable : {img_path}")
            sys.exit(1)
        test_img = Image.open(img_path).convert("RGB")
        print(f"Image chargée : {img_path} ({test_img.size})")
    else:
        # Créer une image synthétique simple
        print("Création image synthétique 256x256...")
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        # Dégradé horizontal
        arr[:, :, 0] = np.linspace(0, 255, 256)[None, :]
        arr[:, :, 1] = np.linspace(0, 255, 256)[:, None]
        arr[:, :, 2] = 128
        test_img = Image.fromarray(arr)

    print("Estimation en cours...")
    try:
        estimator = DepthEstimator(model_size="small")
        depth     = estimator.estimate(test_img)

        print(f"Carte de profondeur : shape={depth.shape}, "
              f"min={depth.min():.3f}, max={depth.max():.3f}")

        # Test avec xyz_ref
        xyz_ref = np.random.randn(1000, 3).astype(np.float32)
        xyz_ref[:, 2] = np.random.uniform(1.0, 5.0, 1000)
        mask    = np.zeros((256, 256), dtype=bool)
        mask[64:192, 64:192] = True

        z_world = estimator.estimate_world_z(test_img, xyz_ref, mask)
        print(f"Z monde (masque) : shape={z_world.shape}, "
              f"min={z_world.min():.3f}, max={z_world.max():.3f}")

        print("=== Test OK ===")

    except Exception as e:
        print(f"Erreur : {e}")
        print("(Normal si transformers/torch non installés)")
        print("Test fallback interpolation...")
        xyz_ref  = np.random.randn(1000, 3).astype(np.float32)
        query_xy = np.random.randn(50, 2).astype(np.float32)
        z_vals   = interpolate_z_from_neighbors(query_xy, xyz_ref)
        print(f"Fallback OK : z_vals shape={z_vals.shape}")
        print("=== Test fallback OK ===")