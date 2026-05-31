"""Sous-systeme Extra : gestionnaire de plugins externes pour Fooocus.

Autonome et portable. Un plugin = un depot GitHub avec un fooocus_extra.json.
Le plugin tourne dans SON propre venv (isolation par process), appele en CLI.

Integration cote Fooocus : voir INTEGRATION.md (patch de ~10 lignes sur webui.py
+ un hook d'offload du modele hote).
"""
import os

from . import manifest, runner, installer, registry

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
INSTALL_ROOT = os.path.join(PACKAGE_DIR, "installed")
OUTPUT_DIR = os.path.join(PACKAGE_DIR, "outputs")


def offload_host_models():
    """Decharge les modeles GPU de Fooocus avant un appel plugin lourd.

    Tolerant : no-op si l'API model_management n'est pas trouvee (portabilite
    entre versions de Fooocus). Renvoie True si l'offload a eu lieu.
    """
    try:
        from ldm_patched.modules import model_management
        model_management.unload_all_models()
        return True
    except Exception:
        return False


__all__ = [
    "manifest", "runner", "installer", "registry",
    "PACKAGE_DIR", "INSTALL_ROOT", "OUTPUT_DIR", "offload_host_models",
]
