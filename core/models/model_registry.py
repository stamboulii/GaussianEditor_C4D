"""
core/models/model_registry.py
------------------------------
Registre générique de lazy loading des modèles IA.

Avant : chaque modèle avait sa propre fonction _get_xxx() dans ge_gsplat_server.py,
avec le cache _models = {} global. Maintenant tout passe par ce registre.

Usage :
    from core.models.model_registry import ModelRegistry
    registry = ModelRegistry.get_instance()
    model = registry.get("langsam")
    registry.release("triposplat")   # libère la VRAM
    registry.release_all()
"""

import threading
from typing import Callable, Any, Optional


class ModelRegistry:
    """
    Singleton thread-safe pour le lazy loading et cache des modèles IA.
    Chaque modèle est chargé une seule fois à la première demande,
    puis mis en cache pour les appels suivants.
    """

    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._cache: dict[str, Any]              = {}
        self._loaders: dict[str, Callable]       = {}
        self._lock = threading.Lock()

    def register(self, name: str, loader_fn: Callable) -> None:
        """
        Enregistre une fonction de chargement pour un modèle.

        Args:
            name: identifiant du modèle (ex: "triposplat", "langsam")
            loader_fn: callable sans argument qui retourne le modèle chargé
        """
        with self._lock:
            self._loaders[name] = loader_fn

    def get(self, name: str) -> Any:
        """
        Retourne le modèle, en le chargeant si nécessaire (lazy).

        Args:
            name: identifiant du modèle

        Returns:
            Le modèle chargé

        Raises:
            KeyError: si le modèle n'est pas enregistré
            Exception: si le chargement échoue
        """
        with self._lock:
            if name in self._cache:
                return self._cache[name]

            if name not in self._loaders:
                raise KeyError(f"Modèle '{name}' non enregistré. "
                               f"Modèles disponibles : {list(self._loaders.keys())}")

        # Charger hors du lock pour ne pas bloquer les autres threads
        print(f"[ModelRegistry] Chargement '{name}'...")
        model = self._loaders[name]()

        with self._lock:
            self._cache[name] = model
            print(f"[ModelRegistry] '{name}' prêt")

        return model

    def is_loaded(self, name: str) -> bool:
        """Retourne True si le modèle est déjà en cache."""
        with self._lock:
            return name in self._cache

    def release(self, name: str) -> None:
        """
        Supprime un modèle du cache et libère la VRAM associée.
        Utile après une génération pour récupérer de la mémoire GPU.
        """
        with self._lock:
            if name not in self._cache:
                return
            del self._cache[name]

        try:
            import torch
            torch.cuda.empty_cache()
            print(f"[ModelRegistry] '{name}' libéré, VRAM nettoyée")
        except Exception:
            pass

    def release_all(self) -> None:
        """Libère tous les modèles du cache."""
        with self._lock:
            names = list(self._cache.keys())
            self._cache.clear()

        try:
            import torch
            torch.cuda.empty_cache()
            print(f"[ModelRegistry] Libéré : {names}")
        except Exception:
            pass

    def loaded_models(self) -> list:
        """Retourne la liste des modèles actuellement en cache."""
        with self._lock:
            return list(self._cache.keys())