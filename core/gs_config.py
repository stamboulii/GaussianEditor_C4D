"""
core/gs_config.py
------------------
Configuration centralisée : ports, chemins, détection matérielle.
"""

import os

# ---------------------------------------------------------------------------
# Réseau
# ---------------------------------------------------------------------------

PORT = int(os.environ.get("GSPLAT_PORT", 8086))
HOST = "127.0.0.1"

# ---------------------------------------------------------------------------
# Chemins de scène (entrée)
# ---------------------------------------------------------------------------

PLY_PATH   = os.environ.get("GS_PLY_PATH", "")
COLMAP_DIR = os.environ.get("GS_COLMAP_DIR", "")

# ---------------------------------------------------------------------------
# Cache HuggingFace
# ---------------------------------------------------------------------------

HF_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")


def setup_hf_cache():
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ["HF_HOME"]               = HF_CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(HF_CACHE_DIR, "hub")
    os.environ["TRANSFORMERS_CACHE"]    = os.path.join(HF_CACHE_DIR, "transformers")
    os.environ["HF_HUB_OFFLINE"]        = "0"


# ---------------------------------------------------------------------------
# TripoSplat
# ---------------------------------------------------------------------------

TRIPOSPLAT_DIR = os.environ.get(
    "TRIPOSPLAT_DIR",
    os.path.join(os.path.expanduser("~"), "TripoSplat")
)

TRIPOSPLAT_CKPTS = {
    "ckpt_path":              os.path.join(TRIPOSPLAT_DIR, "ckpts", "diffusion_models", "triposplat_fp16.safetensors"),
    "decoder_path":           os.path.join(TRIPOSPLAT_DIR, "ckpts", "vae", "triposplat_vae_decoder_fp16.safetensors"),
    "dinov3_path":            os.path.join(TRIPOSPLAT_DIR, "ckpts", "clip_vision", "dino_v3_vit_h.safetensors"),
    "flux2_vae_encoder_path": os.path.join(TRIPOSPLAT_DIR, "ckpts", "vae", "flux2-vae.safetensors"),
    "rmbg_path":              os.path.join(TRIPOSPLAT_DIR, "ckpts", "background_removal", "birefnet.safetensors"),
}

TRIPOSPLAT_DEFAULT_NUM_GAUSSIANS = 32768

# ---------------------------------------------------------------------------
# Sortie scène — pointe vers Windows via /mnt/c/ pour accès depuis C4D
# ---------------------------------------------------------------------------

def get_out_path(filename: str) -> str:
    """
    Retourne un chemin de sortie accessible depuis Windows ET WSL2.

    Priorité :
      1. /mnt/c/Users/*/Documents/C4D/ui_result/  ← accessible depuis Windows
      2. ~/Documents/C4D/ui_result/                ← fallback WSL2
      3. tempdir                                    ← dernier recours
    """
    # Chercher les utilisateurs Windows via /mnt/c
    win_users_base = "/mnt/c/Users"
    if os.path.isdir(win_users_base):
        for user in os.listdir(win_users_base):
            if user in ("Public", "Default", "Default User", "All Users"):
                continue
            candidate = os.path.join(
                win_users_base, user, "Documents", "C4D", "ui_result"
            )
            try:
                os.makedirs(candidate, exist_ok=True)
                return os.path.join(candidate, filename)
            except Exception:
                continue

    # Fallback WSL2
    wsl_candidates = [
        os.path.join(os.path.expanduser("~"), "Documents", "C4D", "ui_result"),
        os.path.join(os.path.expanduser("~"), ".gaussianeditor_c4d", "ui_result"),
        os.path.join(os.getcwd(), "ui_result"),
    ]
    for folder in wsl_candidates:
        try:
            os.makedirs(folder, exist_ok=True)
            return os.path.join(folder, filename)
        except Exception:
            continue

    import tempfile
    return os.path.join(tempfile.gettempdir(), filename)


def wsl_to_win(path: str) -> str:
    """Convertit /mnt/c/... -> C:\\..."""
    if not path or not path.startswith("/mnt/"):
        return path
    parts = path[5:].split("/", 1)
    drive = parts[0].upper()
    rest  = parts[1].replace("/", "\\") if len(parts) > 1 else ""
    return f"{drive}:\\{rest}"


# ---------------------------------------------------------------------------
# Détection matérielle
# ---------------------------------------------------------------------------

def detect_device():
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu", 0.0
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return "cuda", vram
    except Exception:
        return "cpu", 0.0


def recommended_num_gaussians(vram_gb: float) -> int:
    if vram_gb <= 0:    return 16384
    if vram_gb <= 4.5:  return 32768
    if vram_gb <= 8:    return 65536
    if vram_gb <= 12:   return 131072
    return 262144


DEVICE, VRAM_GB = detect_device()
USE_FP16 = (DEVICE == "cuda" and VRAM_GB >= 3.5)