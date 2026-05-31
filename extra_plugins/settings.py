"""Persistance legere des reglages du sous-systeme Extra.

Stocke dans extra_plugins/settings.json (gitignore) :
  - enabled : etat de la case "Extra Plugins" (pour rester actif au reboot).
  - plugins[<id>] : { esrgan_dir, params } memorises par plugin.

Aucune dependance a Fooocus. Ecriture best-effort (n'echoue jamais l'UI).
"""
import json
import os
import threading

_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
_lock = threading.Lock()


def _load():
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _save(data):
    with _lock:
        try:
            with open(_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


def get_enabled():
    return bool(_load().get("enabled", False))


def set_enabled(value):
    d = _load()
    d["enabled"] = bool(value)
    _save(d)


def get_plugin(plugin_id):
    return (_load().get("plugins", {}) or {}).get(plugin_id, {}) or {}


def set_plugin(plugin_id, esrgan_dir=None, params=None):
    d = _load()
    plugins = d.setdefault("plugins", {})
    entry = plugins.setdefault(plugin_id, {})
    if esrgan_dir is not None:
        entry["esrgan_dir"] = esrgan_dir
    if params is not None:
        entry["params"] = params
    _save(d)
