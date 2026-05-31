"""Decouverte des plugins Extra installes (un sous-dossier par plugin).

Un plugin installe = un dossier sous install_root contenant fooocus_extra.json.
"""
import os
import shutil

from . import manifest as manifest_mod


def list_plugins(install_root):
    """Liste les plugins installes valides. Renvoie une liste de dicts."""
    out = []
    if not os.path.isdir(install_root):
        return out
    for name in sorted(os.listdir(install_root)):
        d = os.path.join(install_root, name)
        if not os.path.isdir(d) or not manifest_mod.has_manifest(d):
            continue
        try:
            data = manifest_mod.load(d)
        except manifest_mod.ManifestError:
            continue
        out.append({
            "id": data["id"],
            "name": data["name"],
            "version": data.get("version", "?"),
            "dir": d,
            "manifest": data,
        })
    return out


def get_plugin(install_root, plugin_id):
    for p in list_plugins(install_root):
        if p["id"] == plugin_id:
            return p
    return None


def remove_plugin(install_root, plugin_id):
    p = get_plugin(install_root, plugin_id)
    if not p:
        return False
    shutil.rmtree(p["dir"], ignore_errors=True)
    return True
