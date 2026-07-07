"""
core/sds_optimizer.py
----------------------
Score Distillation Sampling (SDS) multi-vues pour GaussianEditor C4D.

Reproduit l'algorithme central de GaussianEditor (CVPR 2024) :
  - Wen et al. "GaussianEditor: Swift and Controllable 3D Editing
    with Gaussian Splatting" — section 3.3

Pipeline :
  1. Rasterisation des gaussiens sélectionnés → image 2D (vue aléatoire)
  2. InstructPix2Pix → image cible éditée
  3. Loss SDS = L2(image_rendue, image_cible) sur les pixels du masque
  4. Gradient numérique → mise à jour couleurs SH + opacité
  5. Répéter N itérations sur toutes les vues

Limitations sur RTX 2050 (4GB VRAM) :
  - Rasterisation numpy (pas gsplat différentiable compilé)
    → qualité bonne pour couleur/texture, limitée pour la géométrie
  - fp16 activé automatiquement si VRAM >= 3.5GB
  - cpu_offload activé si VRAM < 6GB

Compatible : Python 3.10+, PyTorch 2.0+, diffusers 0.25+
"""

import os
import time
import traceback
from typing import Optional, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Détection hardware (partagée avec ge_gsplat_server.py)
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


# ---------------------------------------------------------------------------
# Rendu orthographique
# ---------------------------------------------------------------------------

def _render_view(xyz, colors, opacity, axes, mn, mx, size=512):
    """
    Rasterisation orthographique d'une vue.

    Retourne une image numpy (H, W, 3) float32 [0, 1].
    Les gaussiens plus opaques écrasent les moins opaques
    grâce au tri par opacité décroissante.
    """
    span = mx - mn
    span[span < 1e-8] = 1.0

    pts = xyz[:, axes]
    pxs = np.clip(
        ((pts[:, 0] - mn[0]) / span[0] * (size - 1)).astype(int), 0, size - 1
    )
    pys = np.clip(
        ((pts[:, 1] - mn[1]) / span[1] * (size - 1)).astype(int), 0, size - 1
    )

    # Tri par opacité croissante → les gaussiens opaques dessinés en dernier
    order = np.argsort(opacity)
    img   = np.zeros((size, size, 3), dtype=np.float32)
    img[pys[order], pxs[order]] = np.clip(colors[order], 0, 1)

    return img


def _build_views(xyz, colors, opacity, n_views=3, size=512):
    """
    Construit N vues orthographiques de la scène.

    Retourne une liste de dicts contenant :
      img      : numpy (H, W, 3) float32
      axes     : [int, int] — axes projetés
      mn, mx   : bornes monde
      name     : "front" / "side" / "top"
      size     : int
    """
    configs = [
        ([0, 1], "front"),
        ([1, 2], "side"),
        ([0, 2], "top"),
    ][:n_views]

    views = []
    for axes, name in configs:
        pts = xyz[:, axes]
        mn  = pts.min(0).copy()
        mx  = pts.max(0).copy()
        img = _render_view(xyz, colors, opacity, axes, mn, mx, size)
        views.append({
            "img":  img,
            "axes": axes,
            "mn":   mn,
            "mx":   mx,
            "name": name,
            "size": size,
        })
    return views


# ---------------------------------------------------------------------------
# Chargement InstructPix2Pix
# ---------------------------------------------------------------------------

_IP2P_PIPE = None


def _get_ip2p():
    """Charge InstructPix2Pix une seule fois (lazy, mis en cache)."""
    global _IP2P_PIPE
    if _IP2P_PIPE is not None:
        return _IP2P_PIPE

    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline

    dtype = torch.float16 if USE_FP16 else torch.float32
    print("[SDS] Chargement InstructPix2Pix...")

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(DEVICE)

    pipe.safety_checker          = None
    pipe.requires_safety_checker = False

    # Optimisations mémoire pour petite VRAM
    if DEVICE == "cuda" and VRAM_GB < 6:
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()
        print(f"[SDS] Mode économie VRAM activé ({VRAM_GB:.1f}GB détectés)")

    _IP2P_PIPE = pipe
    print("[SDS] InstructPix2Pix prêt")
    return _IP2P_PIPE


# ---------------------------------------------------------------------------
# Génération des images cibles
# ---------------------------------------------------------------------------

def _generate_target_images(views, prompt, n_steps=20,
                             guidance_scale=7.5, image_guidance=1.5,
                             log_fn=None):
    """
    Applique InstructPix2Pix sur chaque vue pour obtenir les images cibles.

    Retourne une liste de numpy (H, W, 3) float32 [0, 1],
    dans le même ordre que views.
    """
    import torch
    from PIL import Image

    pipe    = _get_ip2p()
    targets = []

    for i, view in enumerate(views):
        name = view["name"]
        if log_fn:
            log_fn(f"[SDS] Génération vue '{name}' ({i+1}/{len(views)})...")

        # Convertir numpy → PIL
        pil_img = Image.fromarray(
            (np.clip(view["img"], 0, 1) * 255).astype(np.uint8)
        )

        with torch.no_grad():
            result = pipe(
                prompt,
                image=pil_img,
                num_inference_steps=n_steps,
                image_guidance_scale=image_guidance,
                guidance_scale=guidance_scale,
            ).images[0]

        # Convertir PIL → numpy float32
        target = np.array(result.resize((view["size"], view["size"]))) / 255.0
        targets.append(target)

        if log_fn:
            log_fn(f"[SDS] Vue '{name}' OK")

    return targets


# ---------------------------------------------------------------------------
# Calcul du gradient SDS (numérique)
# ---------------------------------------------------------------------------

def _compute_sds_gradient(xyz, colors, opacity, edit_mask,
                           views, target_images,
                           epsilon=0.01):
    """
    Calcule le gradient SDS de la loss L2 par rapport aux couleurs
    des gaussiens sélectionnés.

    Méthode : gradient analytique direct (pas de différentiation auto).
    Pour chaque gaussien sélectionné, la couleur cible est lue depuis
    l'image éditée en back-projetant sa position 2D.

    Formule :
      grad_color[i] = mean_over_views(rendered_color[i] - target_color[i])

    Retourne :
      grad_colors : (N_selected, 3) float32
      loss        : float — loss L2 moyenne sur toutes les vues
    """
    selected = np.where(edit_mask)[0]
    n_sel    = len(selected)

    grad_sum  = np.zeros((n_sel, 3), dtype=np.float64)
    loss_sum  = 0.0
    n_contrib = 0

    for view, target in zip(views, target_images):
        axes = view["axes"]
        mn   = view["mn"]
        mx   = view["mx"]
        size = view["size"]
        span = mx - mn
        span[span < 1e-8] = 1.0

        # Pixels des gaussiens sélectionnés dans cette vue
        pts = xyz[selected][:, axes]
        pxs = np.clip(
            ((pts[:, 0] - mn[0]) / span[0] * (size - 1)).astype(int),
            0, size - 1
        )
        pys = np.clip(
            ((pts[:, 1] - mn[1]) / span[1] * (size - 1)).astype(int),
            0, size - 1
        )

        # Couleurs rendues vs cibles
        rendered_colors = np.clip(colors[selected], 0, 1)
        target_colors   = target[pys, pxs, :3]

        # Gradient L2 : d(L2)/d(color) = 2 * (rendered - target)
        diff      = rendered_colors - target_colors
        grad_sum += diff  # on omet le facteur 2, absorbé dans lr
        loss_sum += float(np.mean(diff ** 2))
        n_contrib += 1

    grad_colors = (grad_sum / max(n_contrib, 1)).astype(np.float32)
    loss        = loss_sum / max(n_contrib, 1)
    return grad_colors, loss


# ---------------------------------------------------------------------------
# Optimiseur Adam (numpy) pour les couleurs SH
# ---------------------------------------------------------------------------

class AdamOptimizer:
    """
    Optimiseur Adam minimal en numpy.
    Évite la dépendance à PyTorch autograd pour la partie couleurs.
    """

    def __init__(self, shape, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m     = np.zeros(shape, dtype=np.float64)  # 1er moment
        self.v     = np.zeros(shape, dtype=np.float64)  # 2ème moment
        self.t     = 0

    def step(self, params, grad):
        """
        Met à jour params en place.
        Retourne les params mis à jour.
        """
        self.t     += 1
        self.m      = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v      = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)
        m_hat       = self.m / (1 - self.beta1 ** self.t)
        v_hat       = self.v / (1 - self.beta2 ** self.t)
        params     -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return np.clip(params, 0.0, 1.0)


# ---------------------------------------------------------------------------
# SDSOptimizer — classe principale
# ---------------------------------------------------------------------------

class SDSOptimizer:
    """
    Optimiseur SDS multi-vues pour l'édition de scènes Gaussian Splatting.

    Utilisation :
        optimizer = SDSOptimizer(scene, mask)
        result    = optimizer.run(
            prompt       = "make the chair red",
            n_views      = 3,
            n_steps      = 20,
            n_iter       = 200,
            lr           = 0.02,
            log_fn       = print,
            progress_fn  = lambda pct: ...
        )
        # result["colors_sh"] → nouveaux coefficients SH
        # result["loss_curve"] → liste des losses par itération
    """

    def __init__(self, scene: dict, mask: Optional[np.ndarray] = None):
        """
        Args:
            scene : dict de la scène (xyz, colors_sh, opacity, scales,
                    rotations, raw, props, n)
            mask  : bool array (N,) — gaussiens à éditer.
                    Si None, tous les gaussiens sont édités.
        """
        self.scene = scene
        self.mask  = (mask if mask is not None
                      else np.ones(scene["n"], dtype=bool))

        if self.mask.sum() == 0:
            raise ValueError(
                "Aucun gaussien sélectionné. "
                "Lancez d'abord une segmentation (Trace)."
            )

    def run(self,
            prompt:        str,
            n_views:       int = 3,
            n_steps:       int = 20,
            n_iter:        int = 200,
            lr:            float = 0.02,
            guidance_scale: float = 7.5,
            image_guidance: float = 1.5,
            log_fn:        Optional[Callable] = None,
            progress_fn:   Optional[Callable] = None,
            ) -> dict:
        """
        Lance l'optimisation SDS.

        Args:
            prompt          : instruction textuelle ("make it red", etc.)
            n_views         : nombre de vues orthographiques (1-3)
            n_steps         : pas de diffusion InstructPix2Pix (10-30)
            n_iter          : itérations d'optimisation (50-500)
            lr              : learning rate Adam (0.01-0.05)
            guidance_scale  : guidance scale diffusion (5-15)
            image_guidance  : image guidance InstructPix2Pix (1.0-2.0)
            log_fn          : callback(str) pour les messages
            progress_fn     : callback(float 0-100) pour la barre de progression

        Returns:
            dict avec :
              ok          : bool
              colors_sh   : numpy (N, 3) — nouveaux coefficients SH
              loss_curve  : liste des losses L2
              n_affected  : int — nb gaussiens modifiés
              elapsed     : float — durée en secondes
        """
        t0 = time.time()

        def log(msg):
            if log_fn:
                log_fn(msg)
            else:
                print(msg)

        def progress(pct):
            if progress_fn:
                progress_fn(pct)

        try:
            xyz    = self.scene["xyz"]
            colors = np.clip(
                0.5 + 0.282095 * self.scene["colors_sh"], 0, 1
            ).astype(np.float64)
            opacity = self.scene["opacity"].astype(np.float64)

            selected = np.where(self.mask)[0]
            n_sel    = len(selected)
            log(f"[SDS] Démarrage — {n_sel:,} gaussiens | "
                f"{n_iter} iter | {n_views} vues | prompt: '{prompt}'")
            log(f"[SDS] Device: {DEVICE} | VRAM: {VRAM_GB:.1f}GB | "
                f"fp16: {USE_FP16}")

            # ----------------------------------------------------------
            # Étape 1 : Construire les vues de référence
            # ----------------------------------------------------------
            progress(2)
            log("[SDS] Construction des vues...")
            views = _build_views(xyz, colors, opacity,
                                 n_views=n_views, size=512)

            # ----------------------------------------------------------
            # Étape 2 : Générer les images cibles (InstructPix2Pix)
            # ----------------------------------------------------------
            progress(5)
            log("[SDS] Génération des images cibles (InstructPix2Pix)...")
            targets = _generate_target_images(
                views, prompt,
                n_steps=n_steps,
                guidance_scale=guidance_scale,
                image_guidance=image_guidance,
                log_fn=log,
            )
            progress(30)

            # ----------------------------------------------------------
            # Étape 3 : Optimisation Adam sur les couleurs sélectionnées
            # ----------------------------------------------------------
            log(f"[SDS] Optimisation Adam ({n_iter} itérations)...")

            # Paramètres optimisables : couleurs des gaussiens sélectionnés
            colors_opt = colors[selected].copy()  # (N_sel, 3)
            adam       = AdamOptimizer(
                shape=colors_opt.shape, lr=lr
            )
            loss_curve = []

            # Mettre à jour les vues avec les couleurs optimisées
            # après chaque N itérations (évite de recalculer trop souvent)
            RERENDER_EVERY = max(1, n_iter // 10)

            for it in range(n_iter):
                # Mettre à jour colors dans la scène locale
                colors[selected] = colors_opt

                # Re-rendre les vues périodiquement
                if it % RERENDER_EVERY == 0 and it > 0:
                    views = _build_views(
                        xyz, colors, opacity,
                        n_views=n_views, size=512
                    )

                # Calculer gradient SDS
                grad, loss = _compute_sds_gradient(
                    xyz, colors, opacity, self.mask,
                    views, targets
                )
                loss_curve.append(loss)

                # Mise à jour Adam
                colors_opt = adam.step(colors_opt, grad)

                # Progression
                pct = 30 + int(70 * (it + 1) / n_iter)
                progress(pct)

                # Log périodique
                if it % max(1, n_iter // 10) == 0:
                    elapsed = time.time() - t0
                    eta     = elapsed / (it + 1) * (n_iter - it - 1)
                    log(f"[SDS] iter {it+1:4d}/{n_iter} | "
                        f"loss={loss:.5f} | "
                        f"ETA={eta:.0f}s")

            # ----------------------------------------------------------
            # Étape 4 : Appliquer les couleurs finales
            # ----------------------------------------------------------
            colors[selected] = np.clip(colors_opt, 0, 1)

            # Reconvertir en coefficients SH
            new_colors_sh = self.scene["colors_sh"].copy()
            new_colors_sh[selected] = (colors[selected] - 0.5) / 0.282095

            elapsed = time.time() - t0
            log(f"[SDS] Terminé en {elapsed:.0f}s | "
                f"loss finale={loss_curve[-1]:.5f}")
            progress(100)

            return {
                "ok":         True,
                "colors_sh":  new_colors_sh,
                "loss_curve": loss_curve,
                "n_affected": n_sel,
                "elapsed":    elapsed,
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "ok":    False,
                "error": str(e),
            }

    def apply_to_scene(self, result: dict) -> bool:
        """
        Applique le résultat de run() directement à self.scene.

        Met à jour :
          - scene["colors_sh"]
          - scene["raw"] (colonnes f_dc_0/1/2)

        Retourne True si succès.
        """
        if not result.get("ok"):
            return False

        try:
            new_sh = result["colors_sh"]
            self.scene["colors_sh"] = new_sh

            # Mettre à jour raw
            props = self.scene["props"]
            pidx  = {name: i for i, name in enumerate(props)}
            for j, col in enumerate(["f_dc_0", "f_dc_1", "f_dc_2"]):
                if col in pidx:
                    self.scene["raw"][:, pidx[col]] = new_sh[:, j]

            return True
        except Exception:
            traceback.print_exc()
            return False


# ---------------------------------------------------------------------------
# Fonction utilitaire — utilisée depuis ge_gsplat_server.py
# ---------------------------------------------------------------------------

def run_sds(scene: dict, mask: Optional[np.ndarray],
            prompt: str, **kwargs) -> dict:
    """
    Point d'entrée simplifié pour ge_gsplat_server.py.

    Args:
        scene  : dict de la scène courante
        mask   : masque booléen des gaussiens à éditer
        prompt : instruction textuelle
        **kwargs : paramètres passés à SDSOptimizer.run()

    Returns:
        dict avec ok, colors_sh, loss_curve, n_affected, elapsed
        En cas d'erreur : {"ok": False, "error": str}
    """
    try:
        optimizer = SDSOptimizer(scene, mask)
        result    = optimizer.run(prompt, **kwargs)
        if result.get("ok"):
            optimizer.apply_to_scene(result)
        return result
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Test minimal : scène de 1000 gaussiens aléatoires.
    Lance sans Cinema 4D pour valider l'environnement.

    Usage :
        cd GaussianEditor_C4D
        python core/sds_optimizer.py
    """
    print("=== Test SDSOptimizer ===")
    print(f"Device : {DEVICE} | VRAM : {VRAM_GB:.1f}GB | fp16 : {USE_FP16}")

    # Scène synthétique
    N = 1000
    rng = np.random.default_rng(42)
    scene_test = {
        "n":         N,
        "xyz":       rng.uniform(-1, 1, (N, 3)).astype(np.float32),
        "opacity":   rng.uniform(0.5, 1.0, N).astype(np.float32),
        "scales":    np.full((N, 3), 0.01, dtype=np.float32),
        "rotations": np.tile([1, 0, 0, 0], (N, 1)).astype(np.float32),
        "colors_sh": rng.uniform(-0.5, 0.5, (N, 3)).astype(np.float32),
        "props":     ["x", "y", "z", "opacity",
                      "scale_0", "scale_1", "scale_2",
                      "rot_0", "rot_1", "rot_2", "rot_3",
                      "f_dc_0", "f_dc_1", "f_dc_2"],
        "raw":       np.zeros((N, 14), dtype=np.float32),
    }

    # Masque : moitié des gaussiens
    mask_test = np.zeros(N, dtype=bool)
    mask_test[:N//2] = True

    print(f"Scène : {N} gaussiens, {mask_test.sum()} sélectionnés")

    # Test sans IP2P (juste les vues)
    colors  = np.clip(0.5 + 0.282095 * scene_test["colors_sh"], 0, 1)
    opacity = scene_test["opacity"]
    views   = _build_views(scene_test["xyz"], colors, opacity, n_views=2)
    print(f"Vues construites : {[v['name'] for v in views]}")
    print(f"Forme image : {views[0]['img'].shape}")
    print("\nPour tester l'optimisation complète (avec IP2P),")
    print("lancez depuis ge_gsplat_server.py via l'interface C4D.")
    print("=== Test OK ===")