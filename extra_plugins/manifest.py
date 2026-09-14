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
        raise ManifestError(f"No {MANIFEST_NAME} in {plugin_dir}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ManifestError(f"{MANIFEST_NAME} unreadable: {e}")
    validate(data)
    return data


def validate(data):
    """Verifie les champs requis. Leve ManifestError si invalide."""
    if not isinstance(data, dict):
        raise ManifestError("The manifest must be a JSON object.")
    ver = data.get("manifest_version")
    if ver != SUPPORTED_VERSION:
        raise ManifestError(
            f"manifest_version {ver} not supported (expected {SUPPORTED_VERSION}).")
    for key in ("id", "name", "entry"):
        if not data.get(key):
            raise ManifestError(f"Missing required field: {key}")
    entry = data["entry"]
    if not isinstance(entry.get("command"), list) or not entry["command"]:
        raise ManifestError("entry.command must be a non-empty list.")
    if not entry.get("input_arg"):
        raise ManifestError("entry.input_arg is required (e.g. -i).")
    out = entry.get("output") or {}
    if out.get("mode") != "print_output":
        raise ManifestError(
            "entry.output.mode must be 'print_output' (the only supported contract).")
    for p in data.get("params", []):
        if not p.get("key") or not p.get("arg"):
            raise ManifestError(f"Invalid param (key and arg required): {p}")
        if not p.get("type"):
            raise ManifestError(f"Param without a type: {p.get('key')}")
    _validate_actions(data)
    return True


def _validate_actions(data):
    """custom-24 : bloc optionnel `actions` (plusieurs operations par plugin)."""
    declared = data.get("actions")
    if declared is None:
        return
    if not isinstance(declared, list) or not declared:
        raise ManifestError("actions must be a non-empty list.")
    keys = {p.get("key") for p in data.get("params", [])}
    seen = set()
    for a in declared:
        if not isinstance(a, dict) or not a.get("id") or not a.get("label"):
            raise ManifestError(f"Invalid action (id and label required): {a}")
        if a["id"] in seen:
            raise ManifestError(f"Duplicate action: {a['id']}")
        seen.add(a["id"])
        if not isinstance(a.get("args", []), list):
            raise ManifestError(f"Action {a['id']}: args must be a list.")
        for k in a.get("params") or []:
            if k not in keys:
                raise ManifestError(f"Action {a['id']}: unknown param '{k}'.")
        for ip in a.get("image_params") or []:
            if not isinstance(ip, dict) or not ip.get("key") or not ip.get("arg"):
                raise ManifestError(f"Action {a['id']}: invalid image_param (key and arg required): {ip}")


def actions(data):
    """custom-24 : actions normalisees d'un plugin.

    Sans bloc `actions`, une seule action : l'Upscale historique (tous les params, mode
    serveur autorise), donc un manifeste v1 existant se comporte exactement comme avant.
    Une action avec des `args` ou des `image_params` ne passe pas par le serveur par
    defaut : le serveur de la famille ne sert que /upscale.
    """
    all_keys = [p["key"] for p in data.get("params", [])]
    declared = data.get("actions")
    if not declared:
        return [{"id": "upscale", "label": "Upscale", "args": [], "params": all_keys,
                 "image_params": [], "server": True, "note": ""}]
    out = []
    for a in declared:
        special = bool(a.get("args") or a.get("image_params"))
        out.append({
            "id": a["id"],
            "label": a["label"],
            "args": list(a.get("args") or []),
            "params": list(a["params"]) if a.get("params") is not None else all_keys,
            "image_params": [dict(ip) for ip in (a.get("image_params") or [])],
            "server": bool(a.get("server", not special)),
            "note": str(a.get("note") or ""),
        })
    return out


def default_params(data):
    """Dict key -> valeur par defaut, depuis la section params."""
    out = {}
    for p in data.get("params", []):
        if "default" in p:
            out[p["key"]] = p["default"]
    return out
