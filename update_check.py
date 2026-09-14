"""custom-19 — Mise a jour GitHub au demarrage, jamais par-dessus le travail local.

Remplace l'auto-update d'origine de entry_with_update.py : pygit2 + un
`reset --hard` des qu'une avance rapide etait possible. Un fichier suivi modifie
ici (webui.py retouche, un preset, un .bat) etait ecrase sans un mot a chaque
lancement qui suivait un push sur GitHub.

Porte de crispz (_update_check.py, crispz-klein 1.35.0 / crispz-studio 348b77b).
Bibliotheque standard + git en ligne de commande seulement : ce module tourne
AVANT l'import de launch.py et doit marcher meme si l'environnement est casse.

Regle de securite (celle de git, en plus strict) : une avance rapide conserve les
modifications locales des fichiers qu'elle ne touche pas. On bloque quand un commit
a recuperer touche un fichier modifie ici, quand il AJOUTE un chemin deja present
hors de git (git l'ecraserait sans rien dire, ex. un run_*.bat local), ou quand la
branche a diverge.

Au demarrage (boot) :
  - a jour, hors ligne, pas de git, pas de branche suivie -> on demarre tel quel ;
  - mise a jour sure -> proposee [o/N], N par defaut au bout du delai ;
  - mise a jour bloquee -> on dit pourquoi, rien n'est touche.

Variables d'environnement :
  FOOOCUS_NO_UPDATE_CHECK=1   ne rien chercher
  FOOOCUS_AUTO_UPDATE=1       appliquer sans demander une mise a jour SURE
                              (console non interactive : Colab, service)
  FOOOCUS_UPDATE_TIMEOUT=20   delai du fetch et de la question, en secondes

En ligne de commande :
  python update_check.py           diagnostic : 0 rien a proposer, 10 sure, 11 bloquee
  python update_check.py --guard   0 = l'avance rapide peut tourner, 11 = bloquee
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
try:
    TIMEOUT = int(os.environ.get('FOOOCUS_UPDATE_TIMEOUT', '20') or 20)
except ValueError:
    TIMEOUT = 20
SHOW = 8  # commits listes au plus


def _git(*args, timeout=15):
    """(code, sortie) d'une commande git dans le depot ; (None, raison) si git manque
    ou si la commande depasse son delai (un reseau qui pend ne doit pas bloquer le boot)."""
    try:
        p = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=timeout)
        return p.returncode, ((p.stdout or '') + ((p.stderr or '') if p.returncode else '')).strip()
    except FileNotFoundError:
        return None, 'git not found'
    except subprocess.TimeoutExpired:
        return None, f'no response within {timeout} s'


def _lines(out):
    return [ln.strip() for ln in (out or '').splitlines() if ln.strip()]


def assess(fetch=True):
    """Etat de la mise a jour. status : 'skip', 'uptodate', 'safe' ou 'blocked'.
    Separe de boot() pour les tests."""
    code, out = _git('rev-parse', '--is-inside-work-tree')
    if code is None:
        return {'status': 'skip', 'why': out}
    if code != 0 or out != 'true':
        return {'status': 'skip', 'why': 'this folder is not a git repository'}
    code, up = _git('rev-parse', '--abbrev-ref', '--symbolic-full-name', '@{u}')
    if code != 0 or not up:
        return {'status': 'skip', 'why': 'the current branch does not track a remote branch'}
    if fetch:
        code, out = _git('fetch', '--quiet', up.split('/', 1)[0], timeout=TIMEOUT)
        if code != 0:
            last = _lines(out)[-1][:90] if _lines(out) else 'fetch failed'
            return {'status': 'skip', 'why': f'GitHub unreachable ({last})'}
    _, behind = _git('rev-list', '--count', 'HEAD..@{u}')
    _, ahead = _git('rev-list', '--count', '@{u}..HEAD')
    behind = int(behind) if str(behind).isdigit() else 0
    ahead = int(ahead) if str(ahead).isdigit() else 0
    if behind == 0:
        return {'status': 'uptodate', 'upstream': up, 'ahead': ahead}
    _, log = _git('log', '--oneline', '--no-decorate', f'-{SHOW}', 'HEAD..@{u}')
    info = {'upstream': up, 'behind': behind, 'ahead': ahead, 'log': _lines(log)}
    if ahead:
        return {**info, 'status': 'blocked',
                'why': f'diverged branch: {ahead} local commit(s) not on GitHub'}
    # Ce que les commits a recuperer touchent (un renommage = suppression + ajout).
    _, changed = _git('diff', '--name-only', '--no-renames', 'HEAD', '@{u}')
    _, added = _git('diff', '--name-only', '--no-renames', '--diff-filter=A', 'HEAD', '@{u}')
    changed, added = set(_lines(changed)), set(_lines(added))
    # Travail local : fichiers suivis modifies, indexes ou non.
    _, local = _git('diff', '--name-only', 'HEAD')
    local = set(_lines(local))
    overlap = sorted(changed & local)
    clobber = sorted(p for p in added if os.path.lexists(os.path.join(ROOT, p)))
    info['local'] = sorted(local)
    if overlap or clobber:
        return {**info, 'status': 'blocked', 'overlap': overlap, 'clobber': clobber,
                'why': 'the update would touch local work'}
    return {**info, 'status': 'safe'}


def apply():
    """Avance rapide sur la branche suivie, SANS reset : git conserve les modifications
    locales des fichiers que la mise a jour ne touche pas, et refuse le reste.
    A n'appeler qu'apres un assess() 'safe'. Renvoie (ok, sortie git)."""
    code, out = _git('merge', '--ff-only', '@{u}', timeout=120)
    return code == 0, out


def ask(question, timeout):
    """Question o/N avec delai. None quand aucune console interactive ne peut repondre
    (Colab, service, sortie redirigee) : l'appelant ne met alors rien a jour."""
    try:
        if not sys.stdin or not sys.stdin.isatty():
            return None
    except Exception:
        return None
    print(f'{question} [y/N] (N in {timeout} s) ', end='', flush=True)
    if os.name == 'nt':
        import msvcrt
        end = time.time() + timeout
        while time.time() < end:
            if msvcrt.kbhit():
                ch = msvcrt.getwche()
                print()
                return ch.lower() in ('o', 'y')
            time.sleep(0.05)
        print()
        return False
    import select
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        print()
        return False
    return sys.stdin.readline().strip().lower() in ('o', 'oui', 'y', 'yes')


def _plural(n, word):
    return f'{n} {word}{"s" if n > 1 else ""}'


def report(st):
    """Affiche l'etat pour la console de run.bat."""
    s = st['status']
    if s == 'skip':
        print(f"[Update] Not checked: {st['why']}.")
        return
    if s == 'uptodate':
        extra = f" ({_plural(st['ahead'], 'local commit')} not pushed)" if st.get('ahead') else ''
        print(f'[Update] Up to date{extra}.')
        return
    n = st['behind']
    word = 'new commits' if n > 1 else 'new commit'
    print(f"[Update] {n} {word} on GitHub ({st['upstream']}):")
    for ln in st['log']:
        print(f'           {ln[:100]}')
    if n > len(st['log']):
        print(f"           ... and {n - len(st['log'])} more")
    if s == 'blocked':
        print(f"[Update] BLOCKED: {st['why']}.")
        for p in st.get('overlap', []):
            print(f'           modified here AND by the update: {p}')
        for p in st.get('clobber', []):
            print(f'           present here outside git, added by the update: {p}')
        print("[Update] Nothing was touched. Commit / stash these files to update.")
        return
    kept = st.get('local') or []
    if kept:
        print(f"[Update] {_plural(len(kept), 'file')} modified here that the update "
              f'does not touch: kept as is.')


def boot():
    """Point d'entree de entry_with_update.py. Ne leve jamais : au pire, on demarre
    sans mise a jour. Renvoie le statut final (pour les tests et les logs)."""
    if os.environ.get('FOOOCUS_NO_UPDATE_CHECK', '') == '1':
        print('[Update] Check disabled (FOOOCUS_NO_UPDATE_CHECK=1).')
        return 'disabled'
    st = assess(fetch=True)
    report(st)
    if st['status'] != 'safe':
        return st['status']
    if os.environ.get('FOOOCUS_AUTO_UPDATE', '') == '1':
        yes = True
    else:
        yes = ask('[Update] Update now?', TIMEOUT)
    if not yes:
        hint = (' Non-interactive console: set FOOOCUS_AUTO_UPDATE=1 to apply.'
                if yes is None else '')
        print('[Update] Starting without updating.' + hint)
        return 'declined'
    ok, out = apply()
    if ok:
        print(f"[Update] Update applied ({_plural(st['behind'], 'commit')}).")
        return 'updated'
    print('[Update] Fast-forward FAILED, nothing changed:')
    for ln in _lines(out)[-6:]:
        print(f'           {ln}')
    return 'failed'


def main(argv):
    try:
        sys.stdout.reconfigure(errors='replace')  # console cmd : pas d'UTF-8 garanti
    except Exception:
        pass
    guard = '--guard' in argv
    if not guard and os.environ.get('FOOOCUS_NO_UPDATE_CHECK', '') == '1':
        print('[Update] Check disabled (FOOOCUS_NO_UPDATE_CHECK=1).')
        return 0
    st = assess(fetch=True)
    report(st)
    if st['status'] == 'blocked':
        return 11
    if st['status'] == 'safe':
        return 0 if guard else 10
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
