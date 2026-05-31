"""Lecture et validation du manifeste d'un plugin Extra (fooocus_extra.json).

Le manifeste est la carte d'identite qu'un depot expose pour etre integre dans
l'onglet Extra de Fooocus. Voir le manifeste de reference dans le depot crispz.

Aucune dependance a Fooocus ici : module pur, testable seul.
"""
import json
import os

MANIFEST_NAME = "fooocus_extra.json"
SUPPORTED_VERSION = 1


class ManifestError(Exception):
    pass


def manifest_path(plugin_dir):
    return os.path.join(plugin_dir, MANIFEST_NAME)


def has_manifest(plugin_dir):
    return os.path.isfile(manifest_path(plugin_dir))


def load(plugin_dir):
    """Charge et valide le manifeste d'un dossier plugin. Renvoie un dict."""
    path = manifest_path(plugin_dir)
    if not os.path.isfile(path):
        raise ManifestError(f"Pas de {MANIFEST_NAME} dans {plugin_dir}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ManifestError(f"{MANIFEST_NAME} illisible: {e}")
    validate(data)
    return data


def validate(data):
    """Verifie les champs requis. Leve ManifestError si invalide."""
    if not isinstance(data, dict):
        raise ManifestError("Le manifeste doit etre un objet JSON.")
    ver = data.get("manifest_version")
    if ver != SUPPORTED_VERSION:
        raise ManifestError(
            f"manifest_version {ver} non supporte (attendu {SUPPORTED_VERSION}).")
    for key in ("id", "name", "entry"):
        if not data.get(key):
            raise ManifestError(f"Champ requis manquant: {key}")
    entry = data["entry"]
    if not isinstance(entry.get("command"), list) or not entry["command"]:
        raise ManifestError("entry.command doit etre une liste non vide.")
    if not entry.get("input_arg"):
        raise ManifestError("entry.input_arg requis (ex: -i).")
    out = entry.get("output") or {}
    if out.get("mode") != "print_output":
        raise ManifestError(
            "entry.output.mode doit etre 'print_output' (seul contrat supporte).")
    for p in data.get("params", []):
        if not p.get("key") or not p.get("arg"):
            raise ManifestError(f"Param invalide (key et arg requis): {p}")
        if not p.get("type"):
            raise ManifestError(f"Param sans type: {p.get('key')}")
    return True


def default_params(data):
    """Dict key -> valeur par defaut, depuis la section params."""
    out = {}
    for p in data.get("params", []):
        if "default" in p:
            out[p["key"]] = p["default"]
    return out
