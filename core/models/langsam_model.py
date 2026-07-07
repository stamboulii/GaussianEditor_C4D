"""
core/models/langsam_model.py
-----------------------------
Wrapper LangSAM (GroundingDINO + SAM) pour la segmentation 3D par texte.

Extrait de ge_gsplat_server.py → _get_langsam() + _try_trace_langsam().
"""

import traceback
from core.models.model_registry import ModelRegistry
from core.gs_config import DEVICE


def _load_langsam():
    """Charge LangSAM (lazy). ~2GB, long au premier chargement."""
    from lang_sam import LangSAM
    return LangSAM()


def register_langsam() -> None:
    """Enregistre LangSAM dans le ModelRegistry."""
    ModelRegistry.get_instance().register("langsam", _load_langsam)


def get_langsam():
    """Retourne l'instance LangSAM, en la chargeant si nécessaire."""
    return ModelRegistry.get_instance().get("langsam")