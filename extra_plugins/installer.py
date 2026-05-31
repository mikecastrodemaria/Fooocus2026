"""Installation full-auto d'un plugin Extra depuis une URL GitHub.

Etapes : git clone -> lecture du manifeste -> creation du venv -> install des
deps selon la strategie d'environnement declaree (fresh_venv par defaut, ou
reuse_python avec --system-site-packages).

L'isolation par venv est ce qui permet a crispz (torch 2.7 cu128) de ne jamais
polluer l'environnement Fooocus.
"""
import os
import re
import shlex
import shutil
import stat
import subprocess

from . import manifest as manifest_mod
from . import runner


def _repo_name_from_url(url):
    name = url.rstrip("/").split("/")[-1]
    return re.sub(r"\.git$", "", name)


def _rmtree_robust(path):
    """Supprime un arbre meme avec des fichiers en lecture seule (objets git
    sous Windows). Leve si la suppression echoue vraiment (process qui tient
    un fichier), au lieu d'echouer en silence."""
    def _onerror(func, p, exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            raise
    shutil.rmtree(path, onerror=_onerror)


def _run(cmd, cwd, log):
    log(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.stdout:
        log(proc.stdout.rstrip())
    if proc.stderr:
        log(proc.stderr.rstrip())
    if proc.returncode != 0:
        raise RuntimeError(f"Echec ({proc.returncode}): {' '.join(cmd)}")
    return proc


def _subst_step(token, plugin_dir, base_python):
    ctx = {
        "venv_pip": runner.venv_pip(plugin_dir),
        "venv_python": runner.venv_python(plugin_dir),
        "plugin_dir": plugin_dir,
        "base_python": base_python or "python",
    }
    for k, v in ctx.items():
        token = token.replace("{" + k + "}", str(v))
    return token


def _expand_cmd(tokens, plugin_dir, base_python):
    """Substitue les placeholders et eclate {base_python} en plusieurs tokens.

    Permet a l'utilisateur de saisir 'py -3.10' (lance launcher) au lieu d'un
    chemin python.exe : le token est splite proprement (chemins avec espaces
    geres via les guillemets).
    """
    out = []
    for t in tokens:
        if t == "{base_python}" and base_python:
            out.extend(shlex.split(base_python, posix=(os.name != "nt")))
        else:
            out.append(_subst_step(t, plugin_dir, base_python))
    return out


def install_from_github(url, install_root, strategy="fresh_venv",
                        base_python=None, log=None, force=False):
    """Clone + installe un plugin. Renvoie le chemin du plugin installe.

    log : callable(str) pour streamer la progression (sinon print).
    """
    log = log or print
    os.makedirs(install_root, exist_ok=True)
    name = _repo_name_from_url(url)
    plugin_dir = os.path.join(install_root, name)

    if os.path.isdir(plugin_dir):
        if not force:
            raise RuntimeError(f"Deja installe: {name}. Utiliser force pour reinstaller.")
        log(f"Suppression de l'install existante: {plugin_dir}")
        try:
            _rmtree_robust(plugin_dir)
        except Exception as e:
            raise RuntimeError(
                f"Impossible de supprimer {plugin_dir} ({e}). Ferme tout process "
                f"qui l'utilise, ou supprime le dossier a la main, puis relance.")

    # 1) Clone
    log(f"== Clone {url} ==")
    _run(["git", "clone", "--depth", "1", url, plugin_dir], cwd=install_root, log=log)

    # 2) Manifeste
    data = manifest_mod.load(plugin_dir)
    log(f"== Manifeste OK : {data['name']} v{data.get('version', '?')} ==")

    # 3) Environnement
    env = data.get("env", {})
    strategies = env.get("strategies", {})
    strat = strategies.get(strategy)
    if not strat:
        raise RuntimeError(f"Strategie '{strategy}' absente du manifeste.")

    if strat.get("create"):
        create = _expand_cmd(strat["create"], plugin_dir, base_python)
        log("== Creation du venv ==")
        _run(create, cwd=plugin_dir, log=log)
        vpy = runner.venv_python(plugin_dir)
        if not os.path.isfile(vpy):
            raise RuntimeError(
                f"Venv non cree ({vpy} absent). Verifie 'Python de base' : mets "
                f"simplement 'py -3.10' (ou un chemin vers python.exe), PAS une "
                f"commande complete avec -c.")

    for step in strat.get("steps", []):
        log(f"== {step.get('name', 'step')} ==")
        if step.get("warn"):
            log(f"[ATTENTION] {step['warn']}")
        cmd = _expand_cmd(step["cmd"], plugin_dir, base_python)
        _run(cmd, cwd=plugin_dir, log=log)

    log(f"== Installe : {name} ==")
    return plugin_dir
