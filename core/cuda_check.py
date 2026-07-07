"""
core/cuda_check.py
------------------
Validation CUDA, GPU et dépendances avant démarrage du backend.

Fix #3 : Gestion gracieuse des dépendances manquantes
Fix #5 : Validation CUDA avant démarrage backend

Usage :
    from core.cuda_check import check_environment, CudaInfo

    info = check_environment(python_executable)
    if not info.cuda_available:
        print(f"Avertissement: {info.cuda_message}")
"""

import os
import sys
import subprocess
import json
from typing import Optional


class EnvironmentInfo:
    """Résultat de la vérification de l'environnement Python externe."""

    def __init__(self):
        self.python_valid    = False
        self.python_version  = ""
        self.cuda_available  = False
        self.cuda_version    = ""
        self.gpu_name        = ""
        self.gpu_memory_gb   = 0.0
        self.torch_version   = ""
        self.has_diffusers   = False
        self.has_gsplat      = False
        self.has_PIL         = False
        self.has_numpy       = False
        self.warnings        = []
        self.errors          = []

    @property
    def is_ready(self) -> bool:
        """True si l'environnement est prêt pour le backend IA."""
        return (self.python_valid and
                self.has_numpy and
                self.has_PIL and
                len(self.errors) == 0)

    @property
    def cuda_message(self) -> str:
        """Message descriptif sur le statut CUDA."""
        if not self.python_valid:
            return "Python introuvable ou invalide"
        if not self.cuda_available:
            return f"CUDA non disponible — les operations IA tourneront sur CPU (tres lent)"
        return f"CUDA {self.cuda_version} — {self.gpu_name} ({self.gpu_memory_gb:.1f} Go)"

    def summary(self) -> str:
        """Résumé lisible de l'environnement."""
        lines = []
        status = "OK" if self.is_ready else "PROBLEMES DETECTES"
        lines.append(f"=== Environnement Python [{status}] ===")
        lines.append(f"Python : {self.python_version}")
        lines.append(f"CUDA   : {self.cuda_message}")
        lines.append(f"torch  : {self.torch_version or 'non installe'}")
        lines.append(f"Packages : numpy={'OK' if self.has_numpy else 'MANQUANT'} | "
                     f"PIL={'OK' if self.has_PIL else 'MANQUANT'} | "
                     f"diffusers={'OK' if self.has_diffusers else 'manquant'} | "
                     f"gsplat={'OK' if self.has_gsplat else 'manquant'}")
        if self.warnings:
            lines.append("Avertissements :")
            for w in self.warnings:
                lines.append(f"  ! {w}")
        if self.errors:
            lines.append("Erreurs :")
            for e in self.errors:
                lines.append(f"  X {e}")
        return "\n".join(lines)


# Script Python injecté dans l'environnement externe pour collecter les infos
_CHECK_SCRIPT = """
import sys
import json

result = {
    "python_version": sys.version,
    "cuda_available": False,
    "cuda_version": "",
    "gpu_name": "",
    "gpu_memory_gb": 0.0,
    "torch_version": "",
    "has_diffusers": False,
    "has_gsplat": False,
    "has_PIL": False,
    "has_numpy": False,
    "warnings": [],
    "errors": [],
}

# numpy
try:
    import numpy as np
    result["has_numpy"] = True
except ImportError:
    result["errors"].append("numpy manquant — pip install numpy")

# PIL
try:
    from PIL import Image
    result["has_PIL"] = True
except ImportError:
    result["errors"].append("Pillow manquant — pip install Pillow")

# torch + CUDA
try:
    import torch
    result["torch_version"] = torch.__version__
    result["cuda_available"] = torch.cuda.is_available()
    if result["cuda_available"]:
        result["cuda_version"] = torch.version.cuda or ""
        result["gpu_name"] = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory
        result["gpu_memory_gb"] = round(mem / 1024**3, 1)
        if result["gpu_memory_gb"] < 4:
            result["warnings"].append(
                f"GPU {result['gpu_name']} a seulement {result['gpu_memory_gb']} Go VRAM. "
                "Minimum recommande: 4 Go pour InstructPix2Pix."
            )
    else:
        result["warnings"].append(
            "CUDA non disponible. Les operations IA utiliseront le CPU (tres lent). "
            "Installez PyTorch avec support CUDA : "
            "pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )
except ImportError:
    result["errors"].append(
        "PyTorch manquant — pip install torch --index-url https://download.pytorch.org/whl/cu121"
    )

# diffusers
try:
    import diffusers
    result["has_diffusers"] = True
except ImportError:
    result["warnings"].append("diffusers manquant — pip install diffusers (requis pour Edit/Add)")

# gsplat
try:
    import gsplat
    result["has_gsplat"] = True
except ImportError:
    result["warnings"].append("gsplat manquant — pip install gsplat")

# requests (pour la communication HTTP)
try:
    import requests
except ImportError:
    result["errors"].append("requests manquant — pip install requests")

print(json.dumps(result))
"""


def check_environment(python_executable: str,
                      timeout: int = 30) -> EnvironmentInfo:
    """
    Vérifie l'environnement Python externe en lançant un subprocess.

    Args:
        python_executable: Chemin vers python.exe de l'env gs_c4d
        timeout:           Délai max en secondes

    Returns:
        EnvironmentInfo avec tous les détails
    """
    info = EnvironmentInfo()

    # Vérifier que le fichier existe
    if not python_executable or not os.path.isfile(python_executable):
        info.errors.append(f"Python introuvable : {python_executable}")
        return info

    try:
        result = subprocess.run(
            [python_executable, "-c", _CHECK_SCRIPT],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace"
        )

        if result.returncode != 0:
            info.errors.append(f"Erreur Python : {result.stderr[:200]}")
            return info

        # Parser le JSON retourné
        data = json.loads(result.stdout.strip())
        info.python_valid   = True
        info.python_version = data.get("python_version", "")
        info.cuda_available = data.get("cuda_available", False)
        info.cuda_version   = data.get("cuda_version", "")
        info.gpu_name       = data.get("gpu_name", "")
        info.gpu_memory_gb  = data.get("gpu_memory_gb", 0.0)
        info.torch_version  = data.get("torch_version", "")
        info.has_diffusers  = data.get("has_diffusers", False)
        info.has_gsplat     = data.get("has_gsplat", False)
        info.has_PIL        = data.get("has_PIL", False)
        info.has_numpy      = data.get("has_numpy", False)
        info.warnings       = data.get("warnings", [])
        info.errors         = data.get("errors", [])

    except subprocess.TimeoutExpired:
        info.errors.append(f"Timeout ({timeout}s) — Python trop lent a demarrer")
    except json.JSONDecodeError as e:
        info.errors.append(f"Reponse Python invalide : {e}")
    except Exception as e:
        info.errors.append(f"Erreur verification : {e}")

    return info


def format_for_dialog(info: EnvironmentInfo) -> str:
    """
    Formate les informations pour un MessageDialog C4D.
    """
    lines = []

    if not info.python_valid:
        lines.append("Python invalide ou introuvable.")
        lines.append("")
        lines.append("Verifiez le chemin dans l'onglet Backend.")
        return "\n".join(lines)

    # GPU / CUDA
    if info.cuda_available:
        lines.append(f"GPU : {info.gpu_name} ({info.gpu_memory_gb} Go)")
        lines.append(f"CUDA : {info.cuda_version}")
    else:
        lines.append("GPU : CUDA non disponible (mode CPU)")

    lines.append(f"PyTorch : {info.torch_version or 'non installe'}")
    lines.append("")

    # Packages
    pkgs = []
    if info.has_numpy:    pkgs.append("numpy OK")
    if info.has_PIL:      pkgs.append("PIL OK")
    if info.has_diffusers:pkgs.append("diffusers OK")
    if info.has_gsplat:   pkgs.append("gsplat OK")
    lines.append("Packages : " + " | ".join(pkgs) if pkgs else "Aucun package trouve")

    # Erreurs
    if info.errors:
        lines.append("")
        lines.append("Erreurs :")
        for e in info.errors:
            lines.append(f"  - {e}")

    # Avertissements
    if info.warnings:
        lines.append("")
        lines.append("Avertissements :")
        for w in info.warnings[:3]:  # Max 3 pour ne pas surcharger
            lines.append(f"  ! {w[:80]}")

    return "\n".join(lines)