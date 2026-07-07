"""
core/models/triposplat_model.py
--------------------------------
Wrapper TripoSplat pour la génération Image → Gaussian Splat.

Fix juillet 2026 :
  - Libère LangSAM avant de charger TripoSplat (VRAM 4GB insuffisante pour les deux)
  - Recharge LangSAM après la génération si nécessaire
"""

import os
import sys
import traceback

from core.gs_config import TRIPOSPLAT_DIR, TRIPOSPLAT_CKPTS, DEVICE, USE_FP16
from core.models.model_registry import ModelRegistry


def _load_triposplat():
    """Charge le pipeline TripoSplat depuis les poids locaux."""
    if TRIPOSPLAT_DIR not in sys.path:
        sys.path.insert(0, TRIPOSPLAT_DIR)

    from triposplat import TripoSplatPipeline

    missing = [k for k, v in TRIPOSPLAT_CKPTS.items() if not os.path.isfile(v)]
    if missing:
        raise FileNotFoundError(
            f"Poids TripoSplat manquants : {missing}\n"
            f"Relancer : voir INSTALLATION.md"
        )

    pipe = TripoSplatPipeline(
        ckpt_path              = TRIPOSPLAT_CKPTS["ckpt_path"],
        decoder_path           = TRIPOSPLAT_CKPTS["decoder_path"],
        dinov3_path            = TRIPOSPLAT_CKPTS["dinov3_path"],
        flux2_vae_encoder_path = TRIPOSPLAT_CKPTS["flux2_vae_encoder_path"],
        rmbg_path              = TRIPOSPLAT_CKPTS["rmbg_path"],
        device                 = DEVICE,
    )
    return pipe


def register_triposplat() -> None:
    ModelRegistry.get_instance().register("triposplat", _load_triposplat)


def generate_from_image(
    image_path: str,
    num_gaussians: int = 32768,
    output_ply_path: str = "",
    release_after: bool = True,
) -> dict:
    """
    Génère un Gaussian Splat depuis une image via TripoSplat.

    Sur VRAM limitée (4GB) :
      - Libère LangSAM avant de charger TripoSplat
      - Libère TripoSplat après génération
    """
    from core.gs_config import get_out_path

    if not os.path.isfile(image_path):
        return {"ok": False, "error": f"Image introuvable : {image_path}"}

    if not output_ply_path:
        base = os.path.splitext(os.path.basename(image_path))[0]
        output_ply_path = get_out_path(f"{base}_triposplat.ply")

    try:
        registry = ModelRegistry.get_instance()

        # ── Libérer LangSAM pour récupérer la VRAM ──────────
        if registry.is_loaded("langsam"):
            print("[TripoSplat] Liberation LangSAM pour liberer VRAM...")
            registry.release("langsam")

        # ── Charger TripoSplat ───────────────────────────────
        pipe = registry.get("triposplat")

        print(f"[TripoSplat] Generation depuis {os.path.basename(image_path)} "
              f"({num_gaussians:,} gaussians)...")

        gaussian, _ = pipe.run(
            image_path,
            num_gaussians=num_gaussians,
            show_progress=True,
        )

        os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
        gaussian.save_ply(output_ply_path)
        print(f"[TripoSplat] PLY sauvegarde : {output_ply_path}")

        # ── Libérer TripoSplat après génération ─────────────
        if release_after:
            registry.release("triposplat")

        return {
            "ok":       True,
            "ply_path": os.path.abspath(output_ply_path),
            "n":        num_gaussians,
        }

    except Exception as e:
        traceback.print_exc()
        return {"ok": False, "error": str(e)}