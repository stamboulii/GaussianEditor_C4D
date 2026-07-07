"""
core/shap_e_client.py
---------------------
Génération de mesh 3D depuis texte ou image via Shap-E (OpenAI, MIT License).
100% local, gratuit, compatible RTX 2050 4GB.

Usage :
    client = ShapEClient()
    mesh_path = client.text_to_mesh("a red pillow")
    mesh_path = client.image_to_mesh("photo.jpg")
"""

import os
import time
import traceback
from typing import Optional, Callable

# Cache des modèles pour éviter les rechargements
_shap_e_models = {}


def _get_device():
    try:
        import torch
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return "cpu"


class ShapEClient:
    """
    Wrapper Shap-E pour génération mesh depuis texte ou image.
    Les modèles (~4GB total) sont téléchargés au premier appel
    et mis en cache dans ~/.cache/shap_e/
    """

    def __init__(self, log_fn: Optional[Callable] = None):
        self.device  = _get_device()
        self.log_fn  = log_fn or print
        self._loaded = False

    def _log(self, msg):
        self.log_fn(msg)

    def _load_models(self):
        """Charge les modèles Shap-E (lazy, mise en cache)."""
        global _shap_e_models
        if _shap_e_models.get("loaded"):
            return

        from shap_e.models.download import load_model, load_config
        from shap_e.diffusion.gaussian_diffusion import diffusion_from_config

        self._log("[ShapE] Chargement transmitter...")
        _shap_e_models["xm"] = load_model("transmitter", device=self.device)

        self._log("[ShapE] Chargement text300M...")
        _shap_e_models["text"] = load_model("text300M", device=self.device)

        self._log("[ShapE] Chargement image300M...")
        _shap_e_models["image"] = load_model("image300M", device=self.device)

        _shap_e_models["diffusion"] = diffusion_from_config(
            load_config("diffusion")
        )
        _shap_e_models["loaded"] = True
        self._log("[ShapE] Modèles prêts")

    def _sample(self, model_key: str, model_kwargs: dict,
                 n_steps: int = 32) -> object:
        """Lance la diffusion et retourne les latents."""
        from shap_e.diffusion.sample import sample_latents

        return sample_latents(
            batch_size      = 1,
            model           = _shap_e_models[model_key],
            diffusion       = _shap_e_models["diffusion"],
            guidance_scale  = 15.0,
            model_kwargs    = model_kwargs,
            progress        = True,
            clip_denoised   = True,
            use_fp16        = (str(self.device) == "cuda"),
            use_karras      = True,
            karras_steps    = n_steps,
            sigma_min       = 1e-3,
            sigma_max       = 160,
            s_churn         = 0,
        )

    def _decode_and_save(self, latents, out_path: str) -> str:
        """Décode les latents en mesh OBJ et sauvegarde."""
        from shap_e.util.notebooks import decode_latent_mesh
        import trimesh

        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

        mesh = decode_latent_mesh(_shap_e_models["xm"], latents[0]).tri_mesh()
        self._log(f"[ShapE] Mesh : {len(mesh.verts)} vertices, {len(mesh.faces)} faces")

        with open(out_path, "w") as f:
            mesh.write_obj(f)

        self._log(f"[ShapE] Sauvegardé : {out_path}")
        return out_path

    def text_to_mesh(self, prompt: str, n_steps: int = 32,
                     out_dir: str = "") -> str:
        """
        Génère un mesh 3D depuis un prompt texte.

        Args:
            prompt  : description de l'objet ("a red velvet pillow")
            n_steps : steps de diffusion (32 = qualité/vitesse équilibrée)
            out_dir : dossier de sortie (défaut : ~/Documents/C4D/ui_result/)

        Returns:
            Chemin vers le fichier .obj généré
        """
        t0 = time.time()
        self._log(f"[ShapE] text_to_mesh : '{prompt}'")

        self._load_models()

        out_dir  = out_dir or _default_out_dir()
        out_path = os.path.join(out_dir, _safe_filename(prompt) + ".obj")

        self._log(f"[ShapE] Diffusion {n_steps} steps...")
        latents = self._sample("text", {"texts": [prompt]}, n_steps)

        path = self._decode_and_save(latents, out_path)
        self._log(f"[ShapE] text_to_mesh terminé en {time.time()-t0:.0f}s")
        return path

    def image_to_mesh(self, image_path: str, n_steps: int = 32,
                      out_dir: str = "") -> str:
        """
        Génère un mesh 3D depuis une image.

        Args:
            image_path : chemin vers l'image (.jpg, .png)
            n_steps    : steps de diffusion
            out_dir    : dossier de sortie

        Returns:
            Chemin vers le fichier .obj généré
        """
        from PIL import Image
        import torch

        t0 = time.time()
        self._log(f"[ShapE] image_to_mesh : {image_path}")

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image introuvable : {image_path}")

        self._load_models()

        # Préparer l'image : RGB 256×256
        img = Image.open(image_path).convert("RGB").resize((256, 256))

        out_dir  = out_dir or _default_out_dir()
        base     = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(out_dir, f"{base}_3d.obj")

        self._log(f"[ShapE] Diffusion {n_steps} steps depuis image...")
        latents = self._sample("image", {"images": [img]}, n_steps)

        path = self._decode_and_save(latents, out_path)
        self._log(f"[ShapE] image_to_mesh terminé en {time.time()-t0:.0f}s")
        return path

    def unload(self):
        """Libère la mémoire GPU."""
        global _shap_e_models
        import torch
        _shap_e_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log("[ShapE] Modèles déchargés")


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _default_out_dir() -> str:
    path = os.path.join(os.path.expanduser("~"), "Documents", "C4D", "ui_result")
    os.makedirs(path, exist_ok=True)
    return path


def _safe_filename(text: str) -> str:
    """Convertit un prompt en nom de fichier valide."""
    import re
    safe = re.sub(r"[^\w\s-]", "", text.lower())
    safe = re.sub(r"[\s]+", "_", safe.strip())
    return safe[:50] or "object"


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    prompt = " ".join(sys.argv[1:]) or "a red pillow"
    print(f"=== Test ShapEClient — '{prompt}' ===")
    client = ShapEClient()
    path   = client.text_to_mesh(prompt, n_steps=32)
    print(f"Mesh généré : {path}")
    print("=== OK ===")