"""
core/models/ip2p_model.py
--------------------------
Wrapper InstructPix2Pix pour l'édition guidée par texte.

Extrait de ge_gsplat_server.py → _get_ip2p().
Gère automatiquement l'optimisation mémoire sur petite VRAM.
"""

import traceback
from core.models.model_registry import ModelRegistry
from core.gs_config import DEVICE, VRAM_GB, USE_FP16


def _load_ip2p():
    """Charge InstructPix2Pix (lazy). ~4GB, optimisé pour petite VRAM."""
    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline

    dtype = torch.float16 if USE_FP16 else torch.float32
    pipe  = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=dtype,
        use_safetensors=True,
    ).to(DEVICE)

    pipe.safety_checker          = None
    pipe.requires_safety_checker = False

    # Optimisations mémoire pour petite VRAM (< 6GB)
    if DEVICE == "cuda" and VRAM_GB < 6:
        pipe.enable_attention_slicing()
        pipe.enable_sequential_cpu_offload()

    return pipe


def register_ip2p() -> None:
    """Enregistre InstructPix2Pix dans le ModelRegistry."""
    ModelRegistry.get_instance().register("ip2p", _load_ip2p)


def get_ip2p():
    """Retourne le pipeline IP2P, en le chargeant si nécessaire."""
    return ModelRegistry.get_instance().get("ip2p")