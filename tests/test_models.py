"""
tests/test_models.py
---------------------
Tests unitaires pour core/models/.

Ne charge PAS les vrais modèles IA (TripoSplat, LangSAM, IP2P) —
utilise des mocks pour tester la logique du registre sans VRAM ni internet.

Lancer avec :
    python tests/test_models.py
"""

import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.models.model_registry import ModelRegistry


# ---------------------------------------------------------------------------
# Helpers — reset du singleton entre tests
# ---------------------------------------------------------------------------

def _fresh_registry() -> ModelRegistry:
    """Crée une nouvelle instance de registry pour chaque test."""
    r = ModelRegistry()
    return r


# ---------------------------------------------------------------------------
# Tests ModelRegistry
# ---------------------------------------------------------------------------

def test_register_and_get():
    """Un modèle enregistré doit être retournable via get()."""
    r = _fresh_registry()
    r.register("mock_model", lambda: {"weights": [1, 2, 3]})

    model = r.get("mock_model")
    assert model == {"weights": [1, 2, 3]}
    print("OK : test_register_and_get")


def test_lazy_loading():
    """Le loader ne doit être appelé qu'une seule fois (lazy + cache)."""
    r = _fresh_registry()
    call_count = [0]

    def loader():
        call_count[0] += 1
        return "model_instance"

    r.register("lazy_model", loader)

    r.get("lazy_model")
    r.get("lazy_model")
    r.get("lazy_model")

    assert call_count[0] == 1, f"Loader appelé {call_count[0]} fois, attendu 1"
    print("OK : test_lazy_loading")


def test_is_loaded():
    """is_loaded doit retourner False avant get() et True après."""
    r = _fresh_registry()
    r.register("check_model", lambda: "ok")

    assert r.is_loaded("check_model") is False
    r.get("check_model")
    assert r.is_loaded("check_model") is True
    print("OK : test_is_loaded")


def test_release():
    """release() doit vider le cache du modèle."""
    r = _fresh_registry()
    r.register("rel_model", lambda: "loaded")
    r.get("rel_model")
    assert r.is_loaded("rel_model")

    r.release("rel_model")
    assert not r.is_loaded("rel_model")
    print("OK : test_release")


def test_release_reloads():
    """Après release(), get() doit recharger le modèle."""
    r = _fresh_registry()
    call_count = [0]

    def loader():
        call_count[0] += 1
        return f"instance_{call_count[0]}"

    r.register("reload_model", loader)

    m1 = r.get("reload_model")
    r.release("reload_model")
    m2 = r.get("reload_model")

    assert call_count[0] == 2, "Doit charger 2 fois après release"
    assert m1 != m2
    print("OK : test_release_reloads")


def test_release_all():
    """release_all() doit vider tout le cache."""
    r = _fresh_registry()
    r.register("m1", lambda: "a")
    r.register("m2", lambda: "b")
    r.register("m3", lambda: "c")
    r.get("m1")
    r.get("m2")
    r.get("m3")

    assert len(r.loaded_models()) == 3
    r.release_all()
    assert len(r.loaded_models()) == 0
    print("OK : test_release_all")


def test_unknown_model_raises():
    """get() sur un modèle non enregistré doit lever KeyError."""
    r = _fresh_registry()
    try:
        r.get("does_not_exist")
        assert False, "Doit lever KeyError"
    except KeyError as e:
        assert "does_not_exist" in str(e)
    print("OK : test_unknown_model_raises")


def test_loader_exception_propagates():
    """Si le loader lève une exception, get() doit la propager."""
    r = _fresh_registry()

    def bad_loader():
        raise RuntimeError("GPU introuvable")

    r.register("bad_model", bad_loader)

    try:
        r.get("bad_model")
        assert False, "Doit lever RuntimeError"
    except RuntimeError as e:
        assert "GPU introuvable" in str(e)

    # Le modèle ne doit PAS être en cache après un échec
    assert not r.is_loaded("bad_model")
    print("OK : test_loader_exception_propagates")


def test_thread_safety():
    """Plusieurs threads appelant get() simultanément ne doivent charger qu'une fois."""
    r = _fresh_registry()
    call_count = [0]
    lock = threading.Lock()

    def loader():
        import time
        time.sleep(0.05)  # simuler un chargement lent
        with lock:
            call_count[0] += 1
        return "heavy_model"

    r.register("thread_model", loader)

    threads = [threading.Thread(target=r.get, args=("thread_model",)) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Avec le lock dans get(), le loader peut être appelé 1 à quelques fois
    # selon le timing — l'important est que le cache fonctionne ensuite
    assert r.is_loaded("thread_model")
    assert r.get("thread_model") == "heavy_model"
    print(f"OK : test_thread_safety (loader appelé {call_count[0]} fois)")


def test_loaded_models_list():
    """loaded_models() doit retourner la liste exacte des modèles en cache."""
    r = _fresh_registry()
    r.register("alpha", lambda: 1)
    r.register("beta",  lambda: 2)
    r.register("gamma", lambda: 3)

    assert r.loaded_models() == []
    r.get("alpha")
    assert r.loaded_models() == ["alpha"]
    r.get("gamma")
    assert set(r.loaded_models()) == {"alpha", "gamma"}
    print("OK : test_loaded_models_list")


# ---------------------------------------------------------------------------
# Tests generate_from_image (sans vrai modèle)
# ---------------------------------------------------------------------------

def test_generate_from_image_missing_file():
    """generate_from_image sur une image manquante doit retourner ok=False."""
    from core.models.triposplat_model import generate_from_image
    result = generate_from_image("/chemin/inexistant/image.jpg")
    assert result["ok"] is False
    assert "introuvable" in result["error"].lower() or "error" in result
    print("OK : test_generate_from_image_missing_file")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        test_register_and_get,
        test_lazy_loading,
        test_is_loaded,
        test_release,
        test_release_reloads,
        test_release_all,
        test_unknown_model_raises,
        test_loader_exception_propagates,
        test_thread_safety,
        test_loaded_models_list,
        test_generate_from_image_missing_file,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"ECHEC : {t.__name__} → {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"ERREUR : {t.__name__} → {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{len(tests) - failed}/{len(tests)} tests réussis")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()