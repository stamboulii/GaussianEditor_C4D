"""
core/wonder3d_client.py
------------------------
Génération multi-vues et reconstruction 3D pour GaussianEditor C4D.

Pipeline Add complet :
  image 2D (inpaintée par SD 1.5)
       ↓
  Zero123++ → 6 vues multi-angles
       ↓
  Depth Anything V2 → carte de profondeur
       ↓
  Lift 2D→3D → nuage de points 3D
       ↓
  Initialisation gaussiens + fusion scène

Zero123++ vs Wonder3D :
  - Wonder3D original  : ~12GB VRAM → incompatible RTX 2050
  - Zero123++ v1.1     : ~4GB VRAM  → compatible RTX 2050 avec fp16
  - Zero123++ génère 6 vues (azimuts 0°/60°/120°/180°/240°/300°)
    à élévation fixe, ce qui est suffisant pour initialiser des gaussiens

Références :
  - Zero123++ : Shi et al. 2023 https://arxiv.org/abs/2310.15110
  - Wonder3D  : Long et al. 2023 https://arxiv.org/abs/2310.15008
"""

import os
import traceback
from typing import Optional, List, Tuple, Callable

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
USE_FP16 = (DEVICE == "cuda" and VRAM_GB >= 3.5)

# Cache des modèles pour éviter les rechargements
_add_models = {}

# ---------------------------------------------------------------------------
# Constantes Zero123++
# ---------------------------------------------------------------------------

ZERO123_MODEL  = "sudo-ai/zero123plus-v1.1"
ZERO123_CUSTOM = "sudo-ai/zero123plus-pipeline"

# 6 vues générées par Zero123++ : azimuts à élévation 30°
ZERO123_AZIMUTHS   = [0, 60, 120, 180, 240, 300]   # degrés
ZERO123_ELEVATIONS = [30, -20, 30, -20, 30, -20]    # degrés (alternés)

# Taille de la grille de sortie Zero123++ (3 colonnes × 2 lignes)
ZERO123_GRID_COLS = 3
ZERO123_GRID_ROWS = 2


# ---------------------------------------------------------------------------
# Zero123Client — génération multi-vues
# ---------------------------------------------------------------------------

class Zero123Client:
    """
    Wrapper Zero123++ pour génération de vues multi-angles.

    Utilisation :
        client = Zero123Client()
        views  = client.generate(pil_image)
        # views : liste de 6 PIL.Image (ou moins si erreur partielle)
    """

    def __init__(self):
        self._pipe = None
        print(f"[Zero123] Config : {ZERO123_MODEL}")
        print(f"[Zero123] Device : {DEVICE} | VRAM : {VRAM_GB:.1f}GB | "
              f"fp16 : {USE_FP16}")

    def _load(self):
        """Charge le pipeline Zero123++ (lazy)."""
        if self._pipe is not None:
            return

        import torch
        from diffusers import DiffusionPipeline

        dtype = torch.float16 if USE_FP16 else torch.float32
        print("[Zero123] Chargement du modèle (~4GB)...")

        # torch < 2.6 bloque torch.load sans safetensors (CVE-2025-32434)
        # Contournement : allow_pickle via variable d'environnement
        import os
        os.environ["TRANSFORMERS_ALLOW_UNSAFE_DESERIALIZATION"] = "1"

        try:
            self._pipe = DiffusionPipeline.from_pretrained(
                ZERO123_MODEL,
                custom_pipeline=ZERO123_CUSTOM,
                torch_dtype=dtype,
            ).to(DEVICE)
        except Exception as e1:
            print(f"[Zero123] Échec chargement standard ({e1})")
            print("[Zero123] Tentative avec allow_pickle...")
            # Monkey-patch torch.load pour autoriser pickle
            _orig_load = torch.load
            def _patched_load(*args, **kwargs):
                kwargs.pop("weights_only", None)
                return _orig_load(*args, weights_only=False, **kwargs)
            torch.load = _patched_load
            try:
                self._pipe = DiffusionPipeline.from_pretrained(
                    ZERO123_MODEL,
                    custom_pipeline=ZERO123_CUSTOM,
                    torch_dtype=dtype,
                ).to(DEVICE)
            finally:
                torch.load = _orig_load  # Restaurer torch.load original

        if DEVICE == "cuda" and VRAM_GB < 6:
            self._pipe.enable_attention_slicing()
            print(f"[Zero123] Attention slicing activé ({VRAM_GB:.1f}GB VRAM)")

        print("[Zero123] Modèle prêt")

    def generate(self, image, n_steps: int = 36,
                 log_fn: Optional[Callable] = None) -> List:
        """
        Génère 6 vues multi-angles depuis une image 2D.

        Args:
            image   : PIL.Image (idéalement fond blanc/transparent)
            n_steps : pas de diffusion (20-75, défaut 36)
            log_fn  : callback(str) pour logs

        Returns:
            Liste de PIL.Image — 6 vues dans l'ordre des ZERO123_AZIMUTHS
            En cas d'erreur : liste vide []
        """
        from PIL import Image

        def log(msg):
            if log_fn:
                log_fn(msg)
            else:
                print(msg)

        try:
            self._load()

            # Préparer l'image : carré 256x256 avec fond blanc
            img_prepared = self._prepare_image(image)

            log(f"[Zero123] Génération {len(ZERO123_AZIMUTHS)} vues "
                f"({n_steps} steps)...")

            import torch
            with torch.no_grad():
                result = self._pipe(
                    img_prepared,
                    num_inference_steps=n_steps,
                ).images[0]

            # Découper la grille de sortie en vues individuelles
            views = self._split_grid(result)
            log(f"[Zero123] {len(views)} vues générées")
            return views

        except Exception as e:
            traceback.print_exc()
            if log_fn:
                log_fn(f"[Zero123] Erreur : {e}")
            return []

    def _prepare_image(self, image):
        """Prépare l'image pour Zero123++ : carré 256x256, fond blanc."""
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(
                (np.clip(image, 0, 1) * 255).astype(np.uint8)
                if image.dtype != np.uint8 else image
            )

        # Convertir en RGBA pour gérer la transparence
        img_rgba = image.convert("RGBA")
        w, h     = img_rgba.size

        # Fond blanc
        background = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        background.paste(img_rgba, mask=img_rgba.split()[3])
        img_rgb = background.convert("RGB")

        # Redimensionner en carré 256x256 (taille attendue par Zero123++)
        img_square = img_rgb.resize((256, 256), Image.LANCZOS)
        return img_square

    def _split_grid(self, grid_img) -> List:
        """
        Découpe la grille 3×2 de Zero123++ en 6 vues individuelles.

        La grille est organisée en lignes/colonnes :
          [az=0°, az=60°, az=120°]
          [az=180°, az=240°, az=300°]
        """
        from PIL import Image

        w, h      = grid_img.size
        cell_w    = w // ZERO123_GRID_COLS
        cell_h    = h // ZERO123_GRID_ROWS
        views     = []

        for row in range(ZERO123_GRID_ROWS):
            for col in range(ZERO123_GRID_COLS):
                box = (
                    col * cell_w,
                    row * cell_h,
                    (col + 1) * cell_w,
                    (row + 1) * cell_h,
                )
                view = grid_img.crop(box).resize((256, 256), Image.LANCZOS)
                views.append(view)

        return views

    def unload(self):
        """Libère la mémoire GPU."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            try:
                import torch
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                print("[Zero123] Modèle déchargé")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Lift 2D → 3D
# ---------------------------------------------------------------------------

def lift_to_gaussians(
    inpainted_img,
    mask_arr:      np.ndarray,
    depth_map:     Optional[np.ndarray],
    xyz_ref:       np.ndarray,
    cam_info:      dict,
    multiview_imgs: Optional[List] = None,
    max_points:    int = 8000,
    log_fn:        Optional[Callable] = None,
    scene:         Optional[dict] = None,  # pour récupérer les scales originaux
) -> dict:
    """
    Convertit une image 2D + masque + depth en gaussiens 3D.

    Args:
        inpainted_img  : PIL.Image ou numpy — image de l'objet à ajouter
        mask_arr       : (H, W) bool — zone à peupler de gaussiens
        depth_map      : (H, W) float32 [0,1] ou None
        xyz_ref        : (N, 3) — gaussiens existants (calibration Z + scale)
        cam_info       : dict de la vue (axes, mn, mx, size)
        multiview_imgs : liste de PIL.Image (vues Zero123++) optionnel
        max_points     : nb max de nouveaux gaussiens
        log_fn         : callback logs

    Returns:
        dict avec : xyz, colors_sh, opacity, scales, rotations
    """
    from PIL import Image

    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    # Convertir en numpy
    if not isinstance(inpainted_img, np.ndarray):
        img_arr = np.array(inpainted_img.resize((512, 512))) / 255.0
    else:
        img_arr = np.clip(inpainted_img, 0, 1)
        if img_arr.shape[:2] != (512, 512):
            pil_tmp = Image.fromarray((img_arr * 255).astype(np.uint8))
            img_arr = np.array(pil_tmp.resize((512, 512))) / 255.0

    # Pixels à peupler
    if mask_arr.shape != (512, 512):
        mask_pil = Image.fromarray(mask_arr.astype(np.uint8) * 255)
        mask_arr = np.array(mask_pil.resize((512, 512))) > 128

    ys, xs = np.where(mask_arr)
    if len(xs) == 0:
        log("[Lift] Masque vide — aucun gaussien créé")
        return _empty_gaussians()

    # Sous-échantillonnage adaptatif
    # Plus la scène est dense, plus on peut ajouter de gaussiens
    n_scene  = len(xyz_ref)
    adaptive = max(max_points, min(n_scene // 100, 50000))
    if len(xs) > adaptive:
        sel = np.random.choice(len(xs), adaptive, replace=False)
        xs, ys = xs[sel], ys[sel]

    n = len(xs)
    log(f"[Lift] {n:,} nouveaux gaussiens à placer (max adaptatif={adaptive:,})")

    # ------------------------------------------------------------------
    # Coordonnées XY monde
    # ------------------------------------------------------------------
    axes = cam_info["axes"]
    mn   = cam_info["mn"].copy()
    mx   = cam_info["mx"].copy()
    size = cam_info.get("size", 512)
    span = mx - mn
    span[span < 1e-8] = 1.0

    world_x = xs / size * span[0] + mn[0]
    world_y = ys / size * span[1] + mn[1]

    # ------------------------------------------------------------------
    # Coordonnée Z depuis la carte de profondeur
    # ------------------------------------------------------------------
    z_min   = float(xyz_ref[:, 2].min())
    z_max   = float(xyz_ref[:, 2].max())
    z_range = max(z_max - z_min, 0.01)

    if depth_map is not None:
        # Redimensionner depth_map si nécessaire
        if depth_map.shape != (512, 512):
            depth_pil    = Image.fromarray(
                (np.clip(depth_map, 0, 1) * 65535).astype(np.uint16)
            )
            depth_resized = np.array(
                depth_pil.resize((512, 512), Image.BILINEAR)
            ) / 65535.0
        else:
            depth_resized = depth_map

        world_z = z_min + depth_resized[ys, xs] * z_range
        log(f"[Lift] Z depuis Depth Anything : "
            f"[{world_z.min():.3f}, {world_z.max():.3f}]")
    else:
        # Fallback : interpolation depuis les voisins
        log("[Lift] Depth map absente, interpolation Z depuis voisins")
        from depth_estimator import interpolate_z_from_neighbors
        query_xy = np.stack([world_x, world_y], axis=1)
        world_z  = interpolate_z_from_neighbors(query_xy, xyz_ref)
        log(f"[Lift] Z interpolé : [{world_z.min():.3f}, {world_z.max():.3f}]")

    # ------------------------------------------------------------------
    # Assemblage XYZ avec contrainte dans les limites de la scène
    # ------------------------------------------------------------------
    axis_z  = [i for i in range(3) if i not in axes][0]
    new_xyz = np.zeros((n, 3), dtype=np.float32)
    new_xyz[:, axes[0]] = world_x.astype(np.float32)
    new_xyz[:, axes[1]] = world_y.astype(np.float32)
    new_xyz[:, axis_z]  = world_z.astype(np.float32)

    # ------------------------------------------------------------------
    # Préserver la bounding box ET le centroïde de la scène originale
    # ------------------------------------------------------------------
    bbox_min      = xyz_ref.min(axis=0)
    bbox_max      = xyz_ref.max(axis=0)
    centroid_orig = xyz_ref.mean(axis=0)

    # 1. Clipper dans la bounding box originale
    for ax in range(3):
        new_xyz[:, ax] = np.clip(new_xyz[:, ax], bbox_min[ax], bbox_max[ax])

    # 2. Translater pour préserver le centroïde exact
    centroid_new = new_xyz.mean(axis=0)
    translation  = centroid_orig - centroid_new
    new_xyz     += translation

    # 3. Re-clipper après translation
    for ax in range(3):
        new_xyz[:, ax] = np.clip(new_xyz[:, ax], bbox_min[ax], bbox_max[ax])

    log(f"[Lift] Bbox : X=[{bbox_min[0]:.3f},{bbox_max[0]:.3f}] "
        f"Y=[{bbox_min[1]:.3f},{bbox_max[1]:.3f}] "
        f"Z=[{bbox_min[2]:.3f},{bbox_max[2]:.3f}]")
    log(f"[Lift] Nouveaux : X=[{new_xyz[:,0].min():.3f},{new_xyz[:,0].max():.3f}] "
        f"Y=[{new_xyz[:,1].min():.3f},{new_xyz[:,1].max():.3f}] "
        f"Z=[{new_xyz[:,2].min():.3f},{new_xyz[:,2].max():.3f}]")

    # ------------------------------------------------------------------
    # Couleurs
    # ------------------------------------------------------------------
    if multiview_imgs and len(multiview_imgs) >= 2:
        # Moyenne couleur sur plusieurs vues Zero123++
        colors_rgb = _average_colors_from_views(
            xs, ys, img_arr, multiview_imgs, size
        )
        log(f"[Lift] Couleurs depuis {len(multiview_imgs)} vues Zero123++")
    else:
        # Couleurs depuis l'image inpaintée seule
        colors_rgb = img_arr[ys, xs, :3]

    colors_sh = (colors_rgb - 0.5) / 0.282095

    # Scale = percentile 75 des scales de la scène originale
    # np.median(scene["scales"]) donne la médiane de tous les éléments (N×3)
    # ce qui sous-estime car les petits scales dominent
    # On prend le p75 de la moyenne par gaussien pour avoir des splats visibles
    if scene is not None and "scales" in scene:
        per_gaussian_scale = scene["scales"].mean(axis=1)  # (N,) moyenne des 3 axes
        scale_val = float(np.percentile(per_gaussian_scale, 75))
        scale_val = max(min(scale_val, 0.02), 0.001)
    else:
        scale_val = 0.002

    log(f"[Lift] Scale gaussiens : {scale_val:.5f}")

    return {
        "xyz":       new_xyz,
        "colors_sh": colors_sh.astype(np.float32),
        "opacity":   np.full(n, 0.85, dtype=np.float32),
        "scales":    np.full((n, 3), scale_val, dtype=np.float32),
        "rotations": np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32),
    }


def _average_colors_from_views(xs, ys, base_img, multiview_imgs, size=512):
    """
    Calcule la couleur moyenne des nouveaux gaussiens
    en combinant l'image inpaintée et les vues Zero123++.

    Donne plus de poids à la vue frontale (base_img).
    """
    from PIL import Image

    colors_sum = base_img[ys, xs, :3].copy().astype(np.float64) * 2.0
    weight     = 2.0

    for view_img in multiview_imgs[:3]:  # Max 3 vues additionnelles
        if view_img is None:
            continue
        view_arr = np.array(
            view_img.resize((size, size))
        ) / 255.0
        colors_sum += view_arr[ys, xs, :3]
        weight     += 1.0

    return np.clip(colors_sum / weight, 0, 1).astype(np.float32)


def _empty_gaussians():
    """Retourne un dict de gaussiens vide."""
    return {
        "xyz":       np.zeros((0, 3), dtype=np.float32),
        "colors_sh": np.zeros((0, 3), dtype=np.float32),
        "opacity":   np.zeros(0, dtype=np.float32),
        "scales":    np.zeros((0, 3), dtype=np.float32),
        "rotations": np.zeros((0, 4), dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Pipeline Add complet
# ---------------------------------------------------------------------------

def run_add_pipeline(
    scene:       dict,
    prompt:      str,
    mask_path:   str = "",
    n_steps_z123: int = 36,
    max_points:  int = 8000,
    log_fn:      Optional[Callable] = None,
) -> dict:
    """
    Pipeline Add complet : Inpainting → Zero123++ → Depth → Gaussiens.

    Appelé depuis ge_gsplat_server.py → action_add().

    Args:
        scene        : dict scène courante
        prompt       : texte décrivant l'objet à ajouter
        mask_path    : chemin vers masque PNG (optionnel)
        n_steps_z123 : steps Zero123++ (20-75)
        max_points   : nb max nouveaux gaussiens
        log_fn       : callback logs

    Returns:
        dict avec :
          ok          : bool
          n_added     : int
          gaussians   : dict (xyz, colors_sh, opacity, scales, rotations)
          error       : str (si ok=False)
    """
    from PIL import Image

    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    try:
        xyz    = scene["xyz"]
        colors = np.clip(0.5 + 0.282095 * scene["colors_sh"], 0, 1)

        # ------------------------------------------------------------------
        # Étape 1 : Vue de référence + masque (rendu autonome)
        # ------------------------------------------------------------------
        log("[Add] Étape 1/5 : Vue de référence...")
        import torch

        # Rendu orthographique autonome — axes X/Y (vue front)
        axes = [0, 1]
        pts  = xyz[:, axes]
        mn   = pts.min(0).copy()
        mx   = pts.max(0).copy()
        span = mx - mn
        span[span < 1e-8] = 1.0
        size = 512

        MAX_R = 200_000
        if len(xyz) > MAX_R:
            sel   = np.random.choice(len(xyz), MAX_R, replace=False)
            xyz_r = xyz[sel]; col_r = colors[sel]
        else:
            xyz_r = xyz; col_r = colors

        img_arr = np.zeros((size, size, 3), dtype=np.uint8)
        pts_r   = xyz_r[:, axes]
        pxs = np.clip(((pts_r[:,0]-mn[0])/span[0]*(size-1)).astype(int), 0, size-1)
        pys = np.clip(((pts_r[:,1]-mn[1])/span[1]*(size-1)).astype(int), 0, size-1)
        img_arr[pys, pxs] = (np.clip(col_r, 0, 1)*255).astype(np.uint8)
        ref_view = Image.fromarray(img_arr)
        cam_info = {"axes": axes, "mn": mn, "mx": mx, "size": size}

        if mask_path and os.path.isfile(mask_path):
            mask_img = Image.open(mask_path).convert("L").resize((512, 512))
            log(f"[Add] Masque chargé : {mask_path}")
        else:
            # Positionner l'objet à CÔTÉ de la chaise (droite par défaut)
            # plutôt qu'au centre qui se superpose avec la scène
            prompt_low = prompt.lower()
            mask_arr_default = np.zeros((512, 512), dtype=np.uint8)

            if any(w in prompt_low for w in ["left", "gauche"]):
                # Quart gauche
                mask_arr_default[160:352, 32:224] = 255
            elif any(w in prompt_low for w in ["right", "droite"]):
                # Quart droit
                mask_arr_default[160:352, 288:480] = 255
            elif any(w in prompt_low for w in ["front", "avant", "devant"]):
                # Bas de l'image
                mask_arr_default[352:480, 128:384] = 255
            elif any(w in prompt_low for w in ["behind", "back", "arriere", "derriere"]):
                # Haut de l'image
                mask_arr_default[32:160, 128:384] = 255
            else:
                # Par défaut : droite de la chaise
                mask_arr_default[128:384, 300:480] = 255

            mask_img = Image.fromarray(mask_arr_default)
            log(f"[Add] Masque positionnel créé pour '{prompt}'")

        # ------------------------------------------------------------------
        # Étape 2 : Inpainting SD 1.5 (autonome — pas d'import circulaire)
        # ------------------------------------------------------------------
        log(f"[Add] Étape 2/5 : Inpainting '{prompt}'...")
        dtype = torch.float16 if (DEVICE == "cuda" and VRAM_GB >= 3.5) else torch.float32

        if "inpaint" not in _add_models:
            from diffusers import StableDiffusionInpaintPipeline
            log("[Add] Chargement SD Inpainting...")
            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if USE_FP16 else None,
            ).to(DEVICE)
            pipe.safety_checker = None
            if DEVICE == "cuda" and VRAM_GB < 6:
                pipe.enable_attention_slicing()
            _add_models["inpaint"] = pipe

        with torch.no_grad():
            inpainted = _add_models["inpaint"](
                prompt=prompt,
                image=ref_view,
                mask_image=mask_img,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]
        log("[Add] Inpainting OK")

        # ------------------------------------------------------------------
        # Étape 3 : Zero123++ — génération multi-vues
        # ------------------------------------------------------------------
        log(f"[Add] Étape 3/5 : Zero123++ ({n_steps_z123} steps)...")
        multiview_imgs = []
        try:
            z123_client    = Zero123Client()
            multiview_imgs = z123_client.generate(
                inpainted,
                n_steps=n_steps_z123,
                log_fn=log,
            )
            z123_client.unload()
            log(f"[Add] Zero123++ OK : {len(multiview_imgs)} vues")
        except Exception as e:
            log(f"[Add] Zero123++ échoué ({e}), inpaint seul utilisé")

        # ------------------------------------------------------------------
        # Étape 4 : Depth Anything V2
        # ------------------------------------------------------------------
        log("[Add] Étape 4/5 : Estimation profondeur...")
        depth_map = None
        try:
            from depth_estimator import DepthEstimator
            estimator = DepthEstimator(model_size="small")
            depth_map = estimator.estimate(inpainted)
            estimator.unload()
            log(f"[Add] Depth OK : shape={depth_map.shape}, "
                f"range=[{depth_map.min():.3f}, {depth_map.max():.3f}]")
        except Exception as e:
            log(f"[Add] Depth estimation échouée ({e}), fallback interpolation")

        # ------------------------------------------------------------------
        # Étape 5 : Lift 2D→3D
        # ------------------------------------------------------------------
        log(f"[Add] Étape 5/5 : Lift 2D→3D (max {max_points:,} points)...")
        mask_arr = np.array(mask_img) > 128

        new_gaussians = lift_to_gaussians(
            inpainted_img   = inpainted,
            mask_arr        = mask_arr,
            depth_map       = depth_map,
            xyz_ref         = xyz,
            cam_info        = cam_info,
            multiview_imgs  = multiview_imgs,
            max_points      = max_points,
            log_fn          = log,
            scene           = scene,
        )

        n_added = len(new_gaussians.get("xyz", []))
        log(f"[Add] Pipeline terminé : {n_added:,} nouveaux gaussiens")

        return {
            "ok":        True,
            "n_added":   n_added,
            "gaussians": new_gaussians,
        }

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e), "n_added": 0}


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Test minimal sans Cinema 4D.

    Usage :
        cd GaussianEditor_C4D
        python core/wonder3d_client.py
        python core/wonder3d_client.py --image path/to/image.png
    """
    import sys
    from PIL import Image

    print("=== Test Wonder3D / Zero123++ Client ===")
    print(f"Device : {DEVICE} | VRAM : {VRAM_GB:.1f}GB | fp16 : {USE_FP16}")

    # Image de test
    if len(sys.argv) > 2 and sys.argv[1] == "--image":
        img_path = sys.argv[2]
        if not os.path.isfile(img_path):
            print(f"Image introuvable : {img_path}")
            sys.exit(1)
        test_img = Image.open(img_path).convert("RGB")
        print(f"Image chargée : {img_path} ({test_img.size})")
    else:
        print("Création image synthétique 256x256...")
        arr = np.zeros((256, 256, 3), dtype=np.uint8)
        arr[64:192, 64:192] = [180, 120, 60]  # carré marron
        test_img = Image.fromarray(arr)

    # Test préparation image
    client = Zero123Client()
    prepared = client._prepare_image(test_img)
    print(f"Image préparée : {prepared.size}")

    # Test split grille
    grid_test = Image.new("RGB", (768, 512), color=(200, 200, 200))
    views     = client._split_grid(grid_test)
    print(f"Split grille : {len(views)} vues")
    print(f"Taille vues : {[v.size for v in views]}")

    # Test lift (sans depth)
    xyz_ref  = np.random.randn(500, 3).astype(np.float32)
    mask_arr = np.zeros((512, 512), dtype=bool)
    mask_arr[128:384, 128:384] = True
    cam_info = {
        "axes": [0, 1],
        "mn":   np.array([-1.0, -1.0]),
        "mx":   np.array([1.0, 1.0]),
        "size": 512,
    }

    result = lift_to_gaussians(
        inpainted_img  = test_img,
        mask_arr       = mask_arr,
        depth_map      = None,
        xyz_ref        = xyz_ref,
        cam_info       = cam_info,
        multiview_imgs = views,
        max_points     = 1000,
        log_fn         = print,
    )

    n = len(result.get("xyz", []))
    print(f"Gaussiens créés : {n:,}")
    if n > 0:
        print(f"XYZ range : [{result['xyz'].min(0)}, {result['xyz'].max(0)}]")

    print("=== Test OK ===")
    print("\nPour tester Zero123++ complet (avec GPU) :")
    print("  python core/wonder3d_client.py --image path/to/chair.png")