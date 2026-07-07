"""
core/models/__init__.py
------------------------
Exports publics du module models.

Usage :
    from core.models import ModelRegistry
    from core.models import generate_from_image
    from core.models import register_all_models
"""

from core.models.model_registry   import ModelRegistry
from core.models.triposplat_model import register_triposplat, generate_from_image
from core.models.langsam_model    import register_langsam, get_langsam
from core.models.ip2p_model       import register_ip2p, get_ip2p


def register_all_models() -> None:
    """
    Enregistre tous les modèles dans le registre.
    À appeler une seule fois au démarrage du serveur.
    Les modèles ne sont PAS chargés ici — seulement enregistrés (lazy).
    """
    register_triposplat()
    register_langsam()
    register_ip2p()
    print("[Models] Tous les modèles enregistrés (lazy — chargés à la demande)")


__all__ = [
    "ModelRegistry",
    "register_all_models",
    "generate_from_image",
    "get_langsam",
    "get_ip2p",
]