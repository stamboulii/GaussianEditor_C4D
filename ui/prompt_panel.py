"""
ui/prompt_panel.py
------------------
Panel de prompt texte réutilisable pour les actions IA.

Fournit un widget C4D avec :
- Champ de saisie du prompt
- Historique des derniers prompts
- Suggestions selon l'action
- Validation avant envoi
"""

try:
    import c4d
    from c4d import gui
    INSIDE_C4D = True
except ImportError:
    INSIDE_C4D = False

# Suggestions de prompts par action
PROMPT_SUGGESTIONS = {
    "trace": [
        "building",
        "sky",
        "ground",
        "tree",
        "road",
        "car",
        "person",
        "wall",
        "window",
        "roof",
    ],
    "edit": [
        "make it look like winter",
        "make it look like summer",
        "make it look like night",
        "make it look like sunset",
        "add snow",
        "make it rainy",
        "change to autumn colors",
        "make it foggy",
    ],
    "add": [
        "a red car",
        "a tree",
        "a bench",
        "a street lamp",
        "a person walking",
        "a bicycle",
        "flowers",
        "a fountain",
    ],
}

# Historique global des prompts (persiste pendant la session)
_prompt_history = []
MAX_HISTORY = 20


def add_to_history(prompt: str):
    """Ajoute un prompt à l'historique."""
    global _prompt_history
    if prompt and prompt not in _prompt_history:
        _prompt_history.insert(0, prompt)
        if len(_prompt_history) > MAX_HISTORY:
            _prompt_history = _prompt_history[:MAX_HISTORY]


def get_history() -> list:
    """Retourne l'historique des prompts."""
    return list(_prompt_history)


def get_suggestions(action: str) -> list:
    """Retourne les suggestions pour une action donnée."""
    return PROMPT_SUGGESTIONS.get(action, [])


def validate_prompt(prompt: str, action: str = "") -> tuple:
    """
    Valide un prompt avant envoi.

    Returns:
        (is_valid: bool, error_message: str)
    """
    prompt = prompt.strip()

    if not prompt:
        return False, "Le prompt ne peut pas etre vide."

    if len(prompt) < 2:
        return False, "Le prompt est trop court (minimum 2 caracteres)."

    if len(prompt) > 500:
        return False, f"Le prompt est trop long ({len(prompt)}/500 caracteres)."

    # Avertissement pour l'édition sans segmentation préalable
    if action == "edit" and len(prompt.split()) < 3:
        return True, "Conseil : utilisez une phrase descriptive comme 'make it look like winter'"

    return True, ""


class PromptValidator:
    """
    Validateur de prompt avec feedback utilisateur.
    Utilisé par main_dialog.py avant d'envoyer une action au backend.
    """

    @staticmethod
    def check_and_warn(prompt: str, action: str) -> bool:
        """
        Vérifie le prompt et affiche un avertissement si nécessaire.
        Retourne True si on peut continuer, False si on doit annuler.
        """
        if not INSIDE_C4D:
            return bool(prompt.strip())

        is_valid, message = validate_prompt(prompt, action)

        if not is_valid:
            gui.MessageDialog(
                f"Prompt invalide :\n{message}",
                c4d.GEMB_OK
            )
            return False

        # Avertissement non bloquant
        if message and not is_valid:
            result = gui.MessageDialog(
                f"{message}\n\nContinuer quand meme ?",
                c4d.GEMB_YESNO
            )
            return result == c4d.GEMB_R_YES

        return True

    @staticmethod
    def suggest(action: str) -> str:
        """
        Affiche une suggestion de prompt pour l'action.
        Retourne le prompt suggéré ou vide si annulé.
        """
        suggestions = get_suggestions(action)
        if not suggestions or not INSIDE_C4D:
            return ""

        # Créer une liste de suggestions
        msg = f"Suggestions pour '{action}' :\n\n"
        for i, s in enumerate(suggestions[:5], 1):
            msg += f"  {i}. {s}\n"
        msg += "\nEntrez votre prompt dans le champ texte."

        gui.MessageDialog(msg, c4d.GEMB_OK)
        return ""