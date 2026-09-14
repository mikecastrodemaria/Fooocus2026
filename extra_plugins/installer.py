"""Installation full-auto d'un plugin Extra depuis une URL GitHub.

Etapes : git clone -> lecture du manifeste -> creation du venv -> install des
deps selon la strategie d'environnement declaree (fresh_venv par defaut, ou
reuse_python avec --system-site-packages).

L'isolation par venv est ce qui permet a crispz (torch 2.7 cu128) de ne jamais
polluer l'environnement Fooocus.
"""
import hashlib
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


# ---------------------------------------------------------------------------
# custom-20 : mise a jour d'un plugin installe.
# Meme garde que la mise a jour de Fooocus (update_check.py, custom-19), en local
# pour garder le sous-systeme autonome : avance rapide seulement, jamais par-dessus
# une modification locale, et deps reinstallees seulement si leur fichier change.
# ---------------------------------------------------------------------------
SHOW = 8


def _git(plugin_dir, *args, timeout=60):
    try:
        p = subprocess.run(["git", *args], cwd=plugin_dir, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        return p.returncode, ((p.stdout or "") + ((p.stderr or "") if p.returncode else "")).strip()
    except FileNotFoundError:
        return None, "git introuvable"
    except subprocess.TimeoutExpired:
        return None, f"pas de reponse en {timeout} s"


def _lines(out):
    return [ln.strip() for ln in (out or "").splitlines() if ln.strip()]


def current_commit(plugin_dir):
    code, out = _git(plugin_dir, "rev-parse", "--short", "HEAD", timeout=10)
    return out if code == 0 else ""


def update_status(plugin_dir, fetch=True):
    """Etat de mise a jour d'un plugin : status 'skip' / 'uptodate' / 'safe' / 'blocked'
    (+ behind, log, overlap, clobber, why), sur le modele de update_check.assess."""
    code, out = _git(plugin_dir, "rev-parse", "--is-inside-work-tree", timeout=10)
    if code is None:
        return {"status": "skip", "why": out}
    if code != 0 or out != "true":
        return {"status": "skip", "why": "le plugin n'est pas un clone git (reinstaller pour le mettre a jour)"}
    code, up = _git(plugin_dir, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", timeout=10)
    if code != 0 or not up:
        return {"status": "skip", "why": "la branche du plugin ne suit aucune branche distante"}
    if fetch:
        code, out = _git(plugin_dir, "fetch", "--quiet", up.split("/", 1)[0], timeout=120)
        if code != 0:
            last = _lines(out)[-1][:120] if _lines(out) else "fetch en echec"
            return {"status": "skip", "why": f"GitHub injoignable ({last})"}
    _, behind = _git(plugin_dir, "rev-list", "--count", "HEAD..@{u}")
    _, ahead = _git(plugin_dir, "rev-list", "--count", "@{u}..HEAD")
    behind = int(behind) if str(behind).isdigit() else 0
    ahead = int(ahead) if str(ahead).isdigit() else 0
    commit = current_commit(plugin_dir)
    if behind == 0:
        return {"status": "uptodate", "upstream": up, "ahead": ahead, "commit": commit}
    _, log = _git(plugin_dir, "log", "--oneline", "--no-decorate", f"-{SHOW}", "HEAD..@{u}")
    info = {"upstream": up, "behind": behind, "ahead": ahead, "log": _lines(log), "commit": commit}
    if ahead:
        return {**info, "status": "blocked",
                "why": f"branche divergente : {ahead} commit(s) local(aux) absent(s) du depot"}
    _, changed = _git(plugin_dir, "diff", "--name-only", "--no-renames", "HEAD", "@{u}")
    _, added = _git(plugin_dir, "diff", "--name-only", "--no-renames", "--diff-filter=A", "HEAD", "@{u}")
    _, local = _git(plugin_dir, "diff", "--name-only", "HEAD")
    changed, added, local = set(_lines(changed)), set(_lines(added)), set(_lines(local))
    overlap = sorted(changed & local)
    clobber = sorted(p for p in added if os.path.lexists(os.path.join(plugin_dir, p)))
    if overlap or clobber:
        return {**info, "status": "blocked", "overlap": overlap, "clobber": clobber,
                "why": "la mise a jour toucherait des fichiers modifies dans le plugin"}
    return {**info, "status": "safe", "local": sorted(local)}


def format_status(st):
    """Lignes lisibles pour le journal du Gestionnaire."""
    s = st["status"]
    head = f"Commit installe : {st['commit']}" if st.get("commit") else None
    lines = [head] if head else []
    if s == "skip":
        return lines + [f"Pas de verification : {st['why']}."]
    if s == "uptodate":
        extra = f" ({st['ahead']} commit(s) local(aux) non pousse(s))" if st.get("ahead") else ""
        return lines + [f"A jour{extra}."]
    lines.append(f"{st['behind']} commit(s) disponible(s) sur {st['upstream']} :")
    lines += [f"  {ln[:110]}" for ln in st["log"]]
    if st["behind"] > len(st["log"]):
        lines.append(f"  ... et {st['behind'] - len(st['log'])} autre(s)")
    if s == "blocked":
        lines.append(f"BLOQUEE : {st['why']}.")
        lines += [f"  modifie dans le plugin ET par la mise a jour : {p}" for p in st.get("overlap", [])]
        lines += [f"  present hors de git, ajoute par la mise a jour : {p}" for p in st.get("clobber", [])]
    else:
        lines.append("Mise a jour sure : clique 'Mettre a jour'.")
    return lines


def _dep_steps(manifest, strategy):
    """Etapes d'env qui installent un fichier de dependances (`... -r <fichier>`) :
    les seules rejouees par une mise a jour (ni creation de venv, ni torch)."""
    strategies = (manifest.get("env") or {}).get("strategies") or {}
    strat = strategies.get(strategy) or strategies.get("fresh_venv") or {}
    out = []
    for step in strat.get("steps", []):
        cmd = step.get("cmd") or []
        if "-r" in cmd:
            i = cmd.index("-r")
            if i + 1 < len(cmd):
                out.append((step, cmd[i + 1]))
    return out


def _file_hash(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha1(f.read()).hexdigest()
    except OSError:
        return None


def update_plugin(plugin_dir, strategy="fresh_venv", base_python=None, log=None, force_deps=False):
    """Met a jour un plugin installe. Leve RuntimeError si la mise a jour est bloquee
    ou echoue (rien n'est touche dans le cas bloque). Renvoie
    {updated, deps, manifest_changed, behind}."""
    log = log or print
    st = update_status(plugin_dir, fetch=True)
    for ln in format_status(st):
        log(ln)
    if st["status"] == "skip":
        raise RuntimeError(st["why"])
    if st["status"] == "blocked":
        raise RuntimeError("mise a jour bloquee, rien n'a ete touche")
    if st["status"] == "uptodate" and not force_deps:
        return {"updated": False, "deps": False, "manifest_changed": False, "behind": 0}

    mpath = manifest_mod.manifest_path(plugin_dir)
    old_manifest = manifest_mod.load(plugin_dir)
    before = {req: _file_hash(os.path.join(plugin_dir, req))
              for _, req in _dep_steps(old_manifest, strategy)}
    manifest_before = _file_hash(mpath)

    updated = False
    if st["status"] == "safe":
        log("== Avance rapide (git merge --ff-only) ==")
        code, out = _git(plugin_dir, "merge", "--ff-only", "@{u}", timeout=300)
        if out:
            log(out)
        if code != 0:
            raise RuntimeError("git a refuse l'avance rapide, rien n'a change")
        updated = True

    new_manifest = manifest_mod.load(plugin_dir)  # leve ManifestError si le nouveau est casse
    ran = False
    for step, req in _dep_steps(new_manifest, strategy):
        req_path = os.path.join(plugin_dir, req)
        if not os.path.isfile(req_path):
            continue
        if force_deps or before.get(req) != _file_hash(req_path):
            why = "force" if force_deps else f"{req} a change"
            log(f"== {step.get('name', 'deps')} ({why}) ==")
            _run(_expand_cmd(step["cmd"], plugin_dir, base_python), cwd=plugin_dir, log=log)
            ran = True
    if not ran:
        log("Dependances inchangees : rien a reinstaller.")
    changed = manifest_before != _file_hash(mpath)
    log(f"== Plugin a jour : {current_commit(plugin_dir)} ==")
    return {"updated": updated, "deps": ran, "manifest_changed": changed,
            "behind": st.get("behind", 0)}
