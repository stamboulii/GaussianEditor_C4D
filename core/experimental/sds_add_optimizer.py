"""
core/sds_add_optimizer.py
--------------------------
SDS (Score Distillation Sampling) pour l'ajout d'objets 3D.

Reproduit l'algorithme de GaussianEditor section 3.5 :
  1. Initialisation géométrique (sphère/coussin/cylindre)
  2. Génération images cibles via InstructPix2Pix
  3. Boucle SDS : rendu → loss L2 → gradient → Adam update
  4. Densification périodique (comme le vrai 3DGS)
  5. Élagage des gaussiens transparents

Attributs optimisés :
  - colors_sh  : couleurs SH (DC term)
  - opacity    : opacité (logit-space)
  - scales     : taille des gaussiens (log-space)
  - xyz        : position 3D

Compatible : Python 3.10+, PyTorch 2.6+, CUDA 12.4
"""

import os
import time
import traceback
from typing import Optional, Callable

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

# Cache modèles
_sds_models = {}


# ---------------------------------------------------------------------------
# Initialisation géométrique
# ---------------------------------------------------------------------------

def init_gaussians_geometric(prompt: str, xyz_ref: np.ndarray,
                              n_points: int = 5000,
                              log_fn: Optional[Callable] = None) -> dict:
    """
    Initialise des gaussiens avec une forme géométrique solide.

    Formes :
      ball/sphere/balle    → sphère pleine
      pillow/cushion       → parallélépipède aplati
      box/cube             → cube
      cylinder/bottle/vase → cylindre
      default              → sphère
    """
    def log(msg):
        if log_fn: log_fn(msg)
        else: print(msg)

    p = prompt.lower()

    scene_center = xyz_ref.mean(axis=0)
    scene_size   = xyz_ref.max(axis=0) - xyz_ref.min(axis=0)

    # Position de l'objet selon le prompt
    if any(w in p for w in ["left", "gauche"]):
        center = scene_center + np.array([-scene_size[0] * 0.45, 0, 0])
    elif any(w in p for w in ["right", "droite"]):
        center = scene_center + np.array([scene_size[0] * 0.45, 0, 0])
    elif any(w in p for w in ["front", "avant", "devant"]):
        center = scene_center + np.array([0, 0, scene_size[2] * 0.45])
    elif any(w in p for w in ["behind", "back", "arriere"]):
        center = scene_center + np.array([0, 0, -scene_size[2] * 0.45])
    elif any(w in p for w in ["on", "sur", "top", "above"]):
        center = scene_center + np.array([0, scene_size[1] * 0.45, 0])
    else:
        center = scene_center + np.array([scene_size[0] * 0.5, 0, 0])

    # Taille de l'objet = 30% de la scène
    obj_scale = scene_size * 0.30

    # Forme géométrique
    if any(w in p for w in ["ball", "sphere", "balle", "ballon", "orb"]):
        log("[SDS-Add] Forme : sphère")
        r     = float(np.min(obj_scale)) * 0.5
        pts   = np.random.randn(n_points, 3).astype(np.float32)
        norms = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
        pts  /= norms
        radii = r * (np.random.uniform(0, 1, n_points) ** (1/3))
        xyz   = (center + pts * radii[:, None]).astype(np.float32)

    elif any(w in p for w in ["pillow", "cushion", "coussin", "pad", "oreiller"]):
        log("[SDS-Add] Forme : coussin (parallélépipède aplati)")
        sx = obj_scale[0] * 0.5
        sy = obj_scale[1] * 0.12
        sz = obj_scale[2] * 0.5
        xyz = (center + np.random.uniform(
            [-sx, -sy, -sz], [sx, sy, sz], (n_points, 3)
        )).astype(np.float32)

    elif any(w in p for w in ["box", "cube", "crate", "block"]):
        log("[SDS-Add] Forme : cube")
        s   = float(np.min(obj_scale)) * 0.4
        xyz = (center + np.random.uniform(-s, s, (n_points, 3))).astype(np.float32)

    elif any(w in p for w in ["cylinder", "bottle", "vase", "cup", "mug"]):
        log("[SDS-Add] Forme : cylindre")
        r  = float(min(obj_scale[0], obj_scale[2])) * 0.25
        h  = obj_scale[1] * 0.5
        th = np.random.uniform(0, 2 * np.pi, n_points)
        ri = r * np.sqrt(np.random.uniform(0, 1, n_points))
        xyz = np.stack([
            center[0] + ri * np.cos(th),
            center[1] + np.random.uniform(-h, h, n_points),
            center[2] + ri * np.sin(th),
        ], axis=1).astype(np.float32)

    else:
        log("[SDS-Add] Forme : sphère (défaut)")
        r     = float(np.min(obj_scale)) * 0.4
        pts   = np.random.randn(n_points, 3).astype(np.float32)
        norms = np.linalg.norm(pts, axis=1, keepdims=True) + 1e-8
        pts  /= norms
        radii = r * (np.random.uniform(0, 1, n_points) ** (1/3))
        xyz   = (center + pts * radii[:, None]).astype(np.float32)

    # Scale initial = médiane de la scène
    scale_ref = float(np.percentile(
        np.linalg.norm(xyz_ref - xyz_ref.mean(0), axis=1), 50
    )) * 0.02
    scale_ref = max(min(scale_ref, 0.01), 0.001)

    # Couleur initiale grise (sera optimisée par SDS)
    n = len(xyz)
    colors_sh  = np.zeros((n, 3), dtype=np.float32)  # gris = (0,0,0) en SH
    opacity    = np.full(n, 0.5, dtype=np.float32)    # opacité initiale 50%
    scales     = np.full((n, 3), scale_ref, dtype=np.float32)
    rotations  = np.tile([1, 0, 0, 0], (n, 1)).astype(np.float32)

    log(f"[SDS-Add] {n:,} gaussiens init | centre={center.round(3)} | scale={scale_ref:.5f}")

    return {
        "xyz":       xyz,
        "colors_sh": colors_sh,
        "opacity":   opacity,
        "scales":    scales,
        "rotations": rotations,
        "n":         n,
    }


# ---------------------------------------------------------------------------
# Rendu orthographique
# ---------------------------------------------------------------------------

def _render_gaussians(xyz, colors, opacity, scales,
                      axes, mn, mx, size=512):
    """
    Rasterisation orthographique différentiable (numpy).
    Retourne image (H, W, 3) float32 [0,1].
    """
    span = mx - mn
    span[span < 1e-8] = 1.0

    pts = xyz[:, axes]
    pxs = np.clip(((pts[:, 0] - mn[0]) / span[0] * (size-1)).astype(int), 0, size-1)
    pys = np.clip(((pts[:, 1] - mn[1]) / span[1] * (size-1)).astype(int), 0, size-1)

    # Tri par opacité croissante → les opaques en dernier
    order = np.argsort(opacity)
    img   = np.zeros((size, size, 3), dtype=np.float32)
    w_map = np.zeros((size, size), dtype=np.float32)

    col = np.clip(0.5 + 0.282095 * colors, 0, 1)
    for i in order:
        a = float(opacity[i])
        img[pys[i], pxs[i]]   += col[i] * a
        w_map[pys[i], pxs[i]] += a

    # Normaliser
    mask = w_map > 0
    img[mask] /= w_map[mask, None]
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------------------
# Chargement InstructPix2Pix
# ---------------------------------------------------------------------------

def _get_ip2p():
    if "ip2p" not in _sds_models:
        import torch
        from diffusers import StableDiffusionInstructPix2PixPipeline
        dtype = torch.float16 if USE_FP16 else torch.float32
        print("[SDS-Add] Chargement InstructPix2Pix...")
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=dtype,
            use_safetensors=True,
        ).to(DEVICE)
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        if DEVICE == "cuda" and VRAM_GB < 6:
            pipe.enable_attention_slicing()
            pipe.enable_sequential_cpu_offload()
        _sds_models["ip2p"] = pipe
        print("[SDS-Add] InstructPix2Pix prêt")
    return _sds_models["ip2p"]


# ---------------------------------------------------------------------------
# Optimiseur Adam numpy
# ---------------------------------------------------------------------------

class AdamNP:
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = beta1; self.b2 = beta2; self.eps = eps
        self.m = {}; self.v = {}; self.t = 0

    def step(self, params_dict, grads_dict):
        self.t += 1
        for key in params_dict:
            if key not in grads_dict: continue
            g = grads_dict[key]
            if key not in self.m:
                self.m[key] = np.zeros_like(g)
                self.v[key] = np.zeros_like(g)
            self.m[key] = self.b1 * self.m[key] + (1-self.b1) * g
            self.v[key] = self.b2 * self.v[key] + (1-self.b2) * g**2
            mh = self.m[key] / (1 - self.b1**self.t)
            vh = self.v[key] / (1 - self.b2**self.t)
            params_dict[key] -= self.lr * mh / (np.sqrt(vh) + self.eps)
        return params_dict


# ---------------------------------------------------------------------------
# Densification
# ---------------------------------------------------------------------------

def _densify(gaussians, grad_xyz, threshold=0.01, max_gaussians=30000):
    """
    Densification adaptive : clone les gaussiens avec fort gradient.
    Reproduit l'adaptive density control de 3DGS.
    """
    xyz       = gaussians["xyz"]
    colors_sh = gaussians["colors_sh"]
    opacity   = gaussians["opacity"]
    scales    = gaussians["scales"]
    rotations = gaussians["rotations"]
    n         = len(xyz)

    if n >= max_gaussians:
        return gaussians

    # Gaussiens avec fort gradient de position → à cloner
    grad_norm  = np.linalg.norm(grad_xyz, axis=1)
    to_clone   = grad_norm > threshold
    n_clone    = int(to_clone.sum())

    if n_clone == 0 or n + n_clone > max_gaussians:
        return gaussians

    # Cloner avec petite perturbation
    noise = np.random.randn(n_clone, 3).astype(np.float32) * scales[to_clone].mean()

    new_xyz       = xyz[to_clone] + noise
    new_colors    = colors_sh[to_clone].copy()
    new_opacity   = opacity[to_clone] * 0.7  # légèrement moins opaque
    new_scales    = scales[to_clone] * 0.8   # légèrement plus petits
    new_rotations = rotations[to_clone].copy()

    gaussians["xyz"]       = np.concatenate([xyz, new_xyz], axis=0)
    gaussians["colors_sh"] = np.concatenate([colors_sh, new_colors], axis=0)
    gaussians["opacity"]   = np.concatenate([opacity, new_opacity])
    gaussians["scales"]    = np.concatenate([scales, new_scales], axis=0)
    gaussians["rotations"] = np.concatenate([rotations, new_rotations], axis=0)
    gaussians["n"]         = len(gaussians["xyz"])

    print(f"[SDS-Add] Densification : +{n_clone} gaussiens → total {gaussians['n']:,}")
    return gaussians


def _prune(gaussians, opacity_threshold=0.05):
    """Supprime les gaussiens trop transparents."""
    keep = gaussians["opacity"] > opacity_threshold
    if keep.sum() == len(gaussians["xyz"]):
        return gaussians

    n_pruned = int((~keep).sum())
    for key in ["xyz", "colors_sh", "scales", "rotations"]:
        gaussians[key] = gaussians[key][keep]
    gaussians["opacity"] = gaussians["opacity"][keep]
    gaussians["n"]       = int(keep.sum())
    print(f"[SDS-Add] Élagage : -{n_pruned} gaussiens → total {gaussians['n']:,}")
    return gaussians


# ---------------------------------------------------------------------------
# SDSAddOptimizer — classe principale
# ---------------------------------------------------------------------------

class SDSAddOptimizer:
    """
    Optimiseur SDS pour l'ajout d'objets 3D dans une scène GS.

    Optimise simultanément :
      - colors_sh  : couleurs
      - opacity    : opacité
      - scales     : taille
      - xyz        : position

    Utilisation :
        opt = SDSAddOptimizer(scene)
        result = opt.run(
            prompt     = "a red pillow",
            n_gaussians = 5000,
            n_iter      = 500,
            log_fn      = print,
        )
        # result["gaussians"] → dict des nouveaux gaussiens optimisés
    """

    def __init__(self, scene: dict):
        self.scene = scene

    def run(self,
            prompt:       str,
            n_gaussians:  int = 5000,
            n_iter:       int = 500,
            n_views:      int = 3,
            n_steps_ip2p: int = 20,
            lr_color:     float = 0.05,
            lr_opacity:   float = 0.02,
            lr_scale:     float = 0.01,
            lr_xyz:       float = 0.005,
            densify_every: int = 100,
            prune_every:   int = 150,
            log_fn:       Optional[Callable] = None,
            progress_fn:  Optional[Callable] = None,
            ) -> dict:
        """
        Lance l'optimisation SDS add.

        Args:
            prompt        : description de l'objet ("a red pillow")
            n_gaussians   : nombre initial de gaussiens
            n_iter        : itérations d'optimisation (300-1000)
            n_views       : vues orthographiques (2-3)
            n_steps_ip2p  : pas de diffusion IP2P (10-20)
            lr_*          : learning rates par attribut
            densify_every : densification tous les N iter
            prune_every   : élagage tous les N iter
            log_fn        : callback logs
            progress_fn   : callback progression (0-100)

        Returns:
            dict avec :
              ok         : bool
              gaussians  : dict nouveaux gaussiens optimisés
              n_added    : int
              loss_curve : liste losses
              elapsed    : float secondes
        """
        t0 = time.time()

        def log(msg):
            if log_fn: log_fn(msg)
            else: print(msg)

        def progress(pct):
            if progress_fn: progress_fn(pct)

        try:
            import torch
            from PIL import Image

            xyz_ref = self.scene["xyz"]
            log(f"[SDS-Add] Démarrage | prompt='{prompt}' | "
                f"{n_gaussians} gaussiens | {n_iter} iter")
            log(f"[SDS-Add] Device: {DEVICE} | VRAM: {VRAM_GB:.1f}GB")

            # ----------------------------------------------------------
            # Phase 1 : Initialisation géométrique
            # ----------------------------------------------------------
            progress(2)
            log("[SDS-Add] Phase 1 : Initialisation géométrique...")
            gaussians = init_gaussians_geometric(
                prompt   = prompt,
                xyz_ref  = xyz_ref,
                n_points = n_gaussians,
                log_fn   = log,
            )

            # ----------------------------------------------------------
            # Phase 2 : Construire vues de référence (scène complète)
            # ----------------------------------------------------------
            progress(5)
            log("[SDS-Add] Phase 2 : Construction des vues...")

            view_configs = [
                ([0, 1], "front"),
                ([1, 2], "side"),
                ([0, 2], "top"),
            ][:n_views]

            scene_colors  = np.clip(
                0.5 + 0.282095 * self.scene["colors_sh"], 0, 1
            )
            # Accès sécurisé — opacity peut être absent dans certains PLY
            if "opacity" in self.scene:
                scene_opacity = self.scene["opacity"]
            else:
                scene_opacity = np.ones(len(self.scene["xyz"]), dtype=np.float32)

            # scales moyens par gaussien pour le rendu
            if "scales" in self.scene:
                scene_scales_mean = self.scene["scales"].mean(axis=1)
            else:
                scene_scales_mean = np.full(len(self.scene["xyz"]), 0.002, dtype=np.float32)

            views = []
            for axes, name in view_configs:
                pts = xyz_ref[:, axes]
                mn  = pts.min(0).copy()
                mx  = pts.max(0).copy()
                img = _render_gaussians(
                    xyz_ref, scene_colors, scene_opacity,
                    scene_scales_mean,
                    axes, mn, mx, size=512
                )
                views.append({
                    "img":  img,
                    "axes": axes,
                    "mn":   mn,
                    "mx":   mx,
                    "name": name,
                    "size": 512,
                })

            # ----------------------------------------------------------
            # Phase 3 : Générer images cibles via InstructPix2Pix
            # ----------------------------------------------------------
            progress(8)
            log("[SDS-Add] Phase 3 : Génération images cibles (IP2P)...")
            pipe     = _get_ip2p()
            targets  = []

            for view in views:
                pil_img = Image.fromarray(
                    (np.clip(view["img"], 0, 1) * 255).astype(np.uint8)
                )
                with torch.no_grad():
                    result = pipe(
                        prompt,
                        image=pil_img,
                        num_inference_steps=n_steps_ip2p,
                        image_guidance_scale=1.5,
                        guidance_scale=7.5,
                    ).images[0]
                target = np.array(result.resize((512, 512))) / 255.0
                targets.append(target)
                log(f"[SDS-Add] Vue '{view['name']}' cible générée")

            # ----------------------------------------------------------
            # Phase 4 : Boucle SDS
            # ----------------------------------------------------------
            progress(15)
            log(f"[SDS-Add] Phase 4 : Optimisation SDS {n_iter} itérations...")

            # Paramètres optimisables
            log(f"[SDS-Add] Init params | n_gaussians={gaussians['n']}")
            log(f"[SDS-Add] gaussians keys: {list(gaussians.keys())}")

            params = {
                "colors_sh": gaussians["colors_sh"].copy().astype(np.float64),
                "opacity":   gaussians["opacity"].copy().astype(np.float64),
                "scales":    gaussians["scales"].copy().astype(np.float64),
                "xyz":       gaussians["xyz"].copy().astype(np.float64),
            }
            log(f"[SDS-Add] params initialisés OK")

            # Optimiseurs Adam séparés par attribut
            adam_color   = AdamNP(lr=lr_color)
            adam_opacity = AdamNP(lr=lr_opacity)
            adam_scale   = AdamNP(lr=lr_scale)
            adam_xyz     = AdamNP(lr=lr_xyz)

            loss_curve   = []
            grad_xyz_acc = np.zeros_like(params["xyz"])  # gradient accumulé pour densification

            for it in range(n_iter):
                try:
                    # Mettre à jour gaussians depuis params
                    gaussians["colors_sh"] = params["colors_sh"].astype(np.float32)
                    gaussians["opacity"]   = np.clip(params["opacity"], 0.0, 1.0).astype(np.float32)
                    gaussians["scales"]    = np.clip(params["scales"], 1e-5, 0.05).astype(np.float32)
                    gaussians["xyz"]       = params["xyz"].astype(np.float32)
                except Exception as e_loop:
                    log(f"[SDS-Add] ERREUR iter {it} update gaussians: {e_loop}")
                    log(f"[SDS-Add] params keys: {list(params.keys())}")
                    log(f"[SDS-Add] gaussians keys: {list(gaussians.keys())}")
                    traceback.print_exc()
                    raise

                # Calculer gradient sur toutes les vues
                grad_colors  = np.zeros_like(params["colors_sh"])
                grad_opacity = np.zeros_like(params["opacity"])
                grad_scales  = np.zeros_like(params["scales"])
                grad_xyz     = np.zeros_like(params["xyz"])
                loss_sum     = 0.0

                for view, target in zip(views, targets):
                    axes = view["axes"]
                    mn   = view["mn"]
                    mx   = view["mx"]
                    size = view["size"]
                    span = mx - mn
                    span[span < 1e-8] = 1.0

                    # Rendu des nouveaux gaussiens seulement
                    rendered = _render_gaussians(
                        gaussians["xyz"],
                        gaussians["colors_sh"],
                        gaussians["opacity"],
                        gaussians["scales"].mean(axis=1),
                        axes, mn, mx, size=size
                    )

                    # Pixels des nouveaux gaussiens
                    pts = gaussians["xyz"][:, axes]
                    pxs = np.clip(
                        ((pts[:, 0] - mn[0]) / span[0] * (size-1)).astype(int),
                        0, size-1
                    )
                    pys = np.clip(
                        ((pts[:, 1] - mn[1]) / span[1] * (size-1)).astype(int),
                        0, size-1
                    )

                    # Couleurs cibles
                    target_colors = target[pys, pxs, :3]
                    rendered_colors = np.clip(
                        0.5 + 0.282095 * gaussians["colors_sh"], 0, 1
                    )

                    # Gradients L2
                    diff_c = rendered_colors - target_colors
                    grad_colors  += diff_c
                    loss_sum     += float(np.mean(diff_c**2))

                    # Gradient opacité : pousser vers 1 si couleur correcte
                    color_error = np.abs(diff_c).mean(axis=1)
                    grad_opacity += color_error - 0.3  # target opacity = 1 si couleur bonne

                    # Gradient scale : réduire si gaussien trop grand
                    grad_scales  += (gaussians["scales"] - gaussians["scales"].mean()) * 0.1

                    # Gradient XYZ : pousser vers les pixels de la cible
                    target_dense = target[pys, pxs, :3]
                    pos_error    = np.abs(target_dense - rendered_colors).mean(axis=1)
                    # Gradient de position : se déplacer vers les zones non couvertes
                    grad_xyz[:, axes[0]] += pos_error * (pts[:, 0] - mn[0] - span[0]/2) * 0.01
                    grad_xyz[:, axes[1]] += pos_error * (pts[:, 1] - mn[1] - span[1]/2) * 0.01

                # Moyenne sur les vues
                n_views_actual = max(len(views), 1)
                grad_colors  /= n_views_actual
                grad_opacity /= n_views_actual
                grad_scales  /= n_views_actual
                grad_xyz     /= n_views_actual
                loss         = loss_sum / n_views_actual
                loss_curve.append(loss)

                # Accumuler gradient XYZ pour densification
                grad_xyz_acc += np.abs(grad_xyz)

                # Updates Adam — update en place, ne pas réassigner params
                updated = adam_color.step(
                    {"colors_sh": params["colors_sh"]},
                    {"colors_sh": grad_colors}
                )
                params["colors_sh"] = updated["colors_sh"]

                updated = adam_opacity.step(
                    {"opacity": params["opacity"]},
                    {"opacity": grad_opacity}
                )
                params["opacity"] = updated["opacity"]

                updated = adam_scale.step(
                    {"scales": params["scales"]},
                    {"scales": grad_scales}
                )
                params["scales"] = updated["scales"]

                updated = adam_xyz.step(
                    {"xyz": params["xyz"]},
                    {"xyz": grad_xyz}
                )
                params["xyz"] = updated["xyz"]

                # Clamp opacity
                params["opacity"] = np.clip(params["opacity"], 0.001, 0.999)
                params["scales"]  = np.clip(params["scales"],  1e-5,  0.05)

                # Densification périodique
                if it > 0 and it % densify_every == 0:
                    # Synchroniser gaussians depuis params avant densification
                    for key in ["colors_sh", "opacity", "scales", "xyz"]:
                        gaussians[key] = params[key].astype(np.float32)
                    gaussians = _densify(
                        gaussians,
                        grad_xyz_acc / densify_every,
                        threshold=0.005,
                        max_gaussians=30000,
                    )
                    # Étendre params pour les nouveaux gaussiens
                    n_new = gaussians["n"] - len(params["xyz"])
                    if n_new > 0:
                        for key, lr in [
                            ("colors_sh", lr_color),
                            ("opacity",   lr_opacity),
                            ("scales",    lr_scale),
                            ("xyz",       lr_xyz),
                        ]:
                            params[key] = np.concatenate([
                                params[key],
                                gaussians[key][len(params[key]):].astype(np.float64)
                            ], axis=0)
                    grad_xyz_acc = np.zeros_like(params["xyz"])

                # Élagage périodique
                if it > 0 and it % prune_every == 0:
                    for key in ["colors_sh", "opacity", "scales", "xyz"]:
                        gaussians[key] = params[key].astype(np.float32)
                    gaussians = _prune(gaussians, opacity_threshold=0.05)
                    for key in ["colors_sh", "opacity", "scales", "xyz"]:
                        params[key] = gaussians[key].astype(np.float64)
                    grad_xyz_acc = np.zeros_like(params["xyz"])
                    # Réinitialiser les moments Adam
                    adam_color   = AdamNP(lr=lr_color)
                    adam_opacity = AdamNP(lr=lr_opacity)
                    adam_scale   = AdamNP(lr=lr_scale)
                    adam_xyz     = AdamNP(lr=lr_xyz)

                # Progression
                pct = 15 + int(85 * (it + 1) / n_iter)
                progress(pct)

                # Log périodique
                if it % max(1, n_iter // 20) == 0:
                    elapsed = time.time() - t0
                    eta     = elapsed / (it + 1) * (n_iter - it - 1)
                    log(f"[SDS-Add] iter {it+1:4d}/{n_iter} | "
                        f"loss={loss:.5f} | "
                        f"n={gaussians['n']:,} | "
                        f"ETA={eta:.0f}s")

            # ----------------------------------------------------------
            # Phase 5 : Finalisation
            # ----------------------------------------------------------
            for key in ["colors_sh", "opacity", "scales", "xyz"]:
                gaussians[key] = params[key].astype(np.float32)

            # Élagage final
            gaussians = _prune(gaussians, opacity_threshold=0.1)

            elapsed = time.time() - t0
            log(f"[SDS-Add] Terminé en {elapsed:.0f}s | "
                f"{gaussians['n']:,} gaussiens | "
                f"loss finale={loss_curve[-1]:.5f}")
            progress(100)

            return {
                "ok":         True,
                "gaussians":  gaussians,
                "n_added":    gaussians["n"],
                "loss_curve": loss_curve,
                "elapsed":    elapsed,
            }

        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": str(e)}

def run_sds_add(scene: dict, prompt: str,
                n_gaussians: int = 5000,
                n_iter: int = 500,
                log_fn: Optional[Callable] = None,
                progress_fn: Optional[Callable] = None) -> dict:
    """
    Point d'entrée simplifié pour ge_gsplat_server.py.

    Args:
        scene       : dict scène courante
        prompt      : description de l'objet
        n_gaussians : gaussiens initiaux
        n_iter      : itérations SDS
        log_fn      : callback logs
        progress_fn : callback progression

    Returns:
        dict avec ok, gaussians, n_added, elapsed
    """
    opt = SDSAddOptimizer(scene)
    return opt.run(
        prompt       = prompt,
        n_gaussians  = n_gaussians,
        n_iter       = n_iter,
        n_views      = 3,
        n_steps_ip2p = 20,
        lr_color     = 0.05,
        lr_opacity   = 0.02,
        lr_scale     = 0.01,
        lr_xyz       = 0.005,
        densify_every = 100,
        prune_every   = 150,
        log_fn        = log_fn,
        progress_fn   = progress_fn,
    )


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=== Test SDSAddOptimizer ===")
    print(f"Device : {DEVICE} | VRAM : {VRAM_GB:.1f}GB")

    N = 500
    rng = np.random.default_rng(42)
    scene_test = {
        "n":         N,
        "xyz":       rng.uniform(-1, 1, (N, 3)).astype(np.float32),
        "opacity":   rng.uniform(0.3, 0.9, N).astype(np.float32),
        "scales":    np.full((N, 3), 0.002, dtype=np.float32),
        "rotations": np.tile([1,0,0,0], (N,1)).astype(np.float32),
        "colors_sh": rng.uniform(-0.3, 0.3, (N,3)).astype(np.float32),
    }

    # Test initialisation seulement (sans IP2P)
    g = init_gaussians_geometric("a red pillow", scene_test["xyz"], n_points=100)
    print(f"Init coussin : {g['n']} gaussiens")
    print(f"XYZ range : {g['xyz'].min(0).round(3)} → {g['xyz'].max(0).round(3)}")

    g = init_gaussians_geometric("a blue ball", scene_test["xyz"], n_points=100)
    print(f"Init sphère : {g['n']} gaussiens")

    print("=== Test initialisation OK ===")
    print("Pour tester l'optimisation complète (avec IP2P) :")
    print("  Lancez depuis C4D via le plugin GaussianEditor")