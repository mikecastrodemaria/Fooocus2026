"""custom-20 — Mode serveur des plugins Extra : le modele reste chaud entre deux appels.

Un plugin qui declare un bloc "server" dans son manifeste (crispz et toute la
famille : `app.py --serve`, FastAPI) peut garder son modele charge entre deux
Upscale. En CLI, chaque appel relance un process et recharge le modele : 66 s a
froid contre 46 s a chaud mesures sur RTX 5090 en 2K (brief crispz, tache D).

Cycle de vie :
  - lance a la premiere demande (Popen, journal dans outputs/<id>/<id>_server.log) ;
  - pret quand GET /health repond ;
  - POST /upscale avec le chemin d'entree + les params du manifeste, par cle ;
  - POST /unload des qu'une generation Fooocus demarre (release_vram_all) : le
    serveur garde son process mais rend la VRAM a SDXL ;
  - arrete a la sortie de Fooocus (atexit), par le bouton Stop server, ou avant
    la mise a jour du plugin.

Un serveur deja actif sur le port (lance a la main) est reutilise et jamais tue.
Le --esrgan-dir est fige au lancement : changer de dossier relance le serveur.

Module pur : bibliotheque standard seulement, aucune dependance a Fooocus.
"""
import atexit
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request

from . import runner

START_TIMEOUT = 300  # s : import torch + FastAPI, le modele se charge au 1er /upscale

_SERVERS = {}  # plugin_id -> _Server
_LOCK = threading.Lock()


class ServerError(RuntimeError):
    """Le mode serveur n'a pas pu servir la demande (l'UI retombe alors sur la CLI)."""


def server_spec(manifest):
    """Le bloc server du manifeste s'il est exploitable (launch = liste non vide), sinon None."""
    s = (manifest or {}).get('server') or {}
    launch = s.get('launch')
    return s if isinstance(launch, list) and launch else None


def _flag_value(tokens, flag, default=None):
    for i, t in enumerate(tokens):
        if t == flag and i + 1 < len(tokens):
            return str(tokens[i + 1])
    return default


def server_url(manifest):
    """URL du serveur deduite de server.port ou des flags --host/--port du launch."""
    spec = server_spec(manifest)
    if not spec:
        return None
    port = spec.get('port') or _flag_value(spec['launch'], '--port')
    if not port:
        return None
    host = _flag_value(spec['launch'], '--host', '127.0.0.1')
    if host in ('0.0.0.0', '::', ''):
        host = '127.0.0.1'
    return f'http://{host}:{port}'


def _http(method, url, payload=None, timeout=10):
    data = json.dumps(payload).encode('utf-8') if payload is not None else None
    headers = {'Content-Type': 'application/json'} if data is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as e:
        detail = ''
        try:
            raw = e.read().decode('utf-8', errors='replace')
            detail = raw
            if raw.strip().startswith('{'):
                detail = json.loads(raw).get('detail', raw)
        except Exception:
            pass
        raise ServerError(f'HTTP {e.code} {e.reason}: {str(detail)[:300]}')
    except (urllib.error.URLError, OSError) as e:
        raise ServerError(f'unreachable ({getattr(e, "reason", e)})')
    if not body.strip():
        return {}
    try:
        return json.loads(body)
    except ValueError:
        raise ServerError(f'non-JSON response: {body[:200]}')


def health(url, timeout=3):
    """Le dict /health si le serveur repond, None sinon. Ne leve jamais."""
    try:
        res = _http('GET', url.rstrip('/') + '/health', timeout=timeout)
        return res if isinstance(res, dict) else {}
    except ServerError:
        return None


def log_tail(path, n=12):
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            return '\n'.join(f.read().splitlines()[-n:])
    except Exception:
        return '(log unreadable)'


class _Server:
    def __init__(self, plugin_id, url, proc=None, log_path=None, esrgan_dir=None, log_file=None):
        self.plugin_id = plugin_id
        self.url = url
        self.proc = proc          # None = serveur externe, reutilise mais jamais tue
        self.log_path = log_path
        self.esrgan_dir = esrgan_dir
        self.log_file = log_file
        self.warm = False         # un /upscale a reussi depuis le dernier /unload

    @property
    def owned(self):
        return self.proc is not None

    def running(self):
        return self.proc is None or self.proc.poll() is None


def _launch_command(plugin, esrgan_dir):
    m, pdir = plugin['manifest'], plugin['dir']
    ctx = {'venv_python': runner.venv_python(pdir), 'plugin_dir': pdir}
    cmd = [runner._subst(t, ctx) for t in server_spec(m)['launch']]
    models_arg = (m.get('entry') or {}).get('models_arg')
    if esrgan_dir and models_arg:
        cmd += [runner._subst(t, {'esrgan_dir': esrgan_dir}) for t in models_arg]
    return cmd


def _terminate(srv):
    if srv.owned and srv.proc.poll() is None:
        srv.proc.terminate()
        try:
            srv.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            srv.proc.kill()
    if srv.log_file is not None:
        try:
            srv.log_file.close()
        except Exception:
            pass
        srv.log_file = None


def ensure(plugin, esrgan_dir=None, log_dir=None, start_timeout=None):
    """URL d'un serveur pret pour ce plugin : reutilise, relance ou demarre.
    Leve ServerError avec la fin du journal si le serveur ne demarre pas."""
    pid = plugin['id']
    url = server_url(plugin['manifest'])
    if not url:
        raise ServerError('the manifest declares no usable server (server block + --port)')
    esrgan_dir = esrgan_dir or None
    with _LOCK:
        srv = _SERVERS.get(pid)
        if srv is not None and (not srv.running() or (srv.owned and srv.esrgan_dir != esrgan_dir)):
            # mort, ou lance avec un autre --esrgan-dir (fige au demarrage) : on relance
            _terminate(srv)
            _SERVERS.pop(pid, None)
            srv = None
        if srv is not None and not srv.owned:
            if health(url) is not None:
                return url
            _SERVERS.pop(pid, None)
            raise ServerError(f'the external server {url} no longer responds')
        if srv is None:
            if health(url) is not None:
                _SERVERS[pid] = _Server(pid, url, esrgan_dir=esrgan_dir)
                print(f'[Extra] Server already running on {url}: reused for {pid}.')
                return url
            cmd = _launch_command(plugin, esrgan_dir)
            log_dir = log_dir or plugin['dir']
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f'{pid}_server.log')
            log_file = open(log_path, 'w', encoding='utf-8', errors='replace')
            kwargs = {}
            if os.name == 'nt':
                kwargs['creationflags'] = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
            try:
                proc = subprocess.Popen(cmd, cwd=plugin['dir'], stdout=log_file,
                                        stderr=subprocess.STDOUT, **kwargs)
            except OSError as e:
                log_file.close()
                raise ServerError(f'could not launch ({e})')
            srv = _Server(pid, url, proc, log_path, esrgan_dir, log_file)
            _SERVERS[pid] = srv
            print(f'[Extra] Server {pid} started on {url} (log: {log_path}).')
    timeout = start_timeout or START_TIMEOUT
    deadline = time.time() + timeout
    while time.time() < deadline:
        if srv.proc.poll() is not None:
            with _LOCK:
                if _SERVERS.get(pid) is srv:
                    _SERVERS.pop(pid, None)
            _terminate(srv)
            raise ServerError(f"the server stopped during startup (code {srv.proc.returncode}):\n"
                              f'{log_tail(srv.log_path)}')
        if health(url) is not None:
            return url
        time.sleep(0.5)
    stop(pid)
    raise ServerError(f'no response from {url}/health within {int(timeout)} s:\n{log_tail(srv.log_path)}')


def upscale(plugin_id, url, payload, timeout=1800):
    """POST /upscale. Renvoie la reponse du serveur + 'was_warm' (modele deja charge)."""
    res = _http('POST', url.rstrip('/') + '/upscale', payload=payload, timeout=timeout)
    if not isinstance(res, dict) or not res.get('output'):
        raise ServerError(f'no file in the response: {str(res)[:200]}')
    srv = _SERVERS.get(plugin_id)
    res['was_warm'] = bool(srv and srv.warm)
    if srv is not None:
        srv.warm = True
    return res


def release_vram_all(timeout=5):
    """POST /unload a chaque serveur qui tient un modele. Appele au debut de chaque
    generation Fooocus. No-op instantane sans serveur. Renvoie les ids liberes."""
    with _LOCK:
        servers = [s for s in _SERVERS.values() if s.warm and s.running()]
    freed = []
    for s in servers:
        try:
            _http('POST', s.url.rstrip('/') + '/unload', payload={}, timeout=timeout)
            s.warm = False
            freed.append(s.plugin_id)
        except ServerError:
            pass
    return freed


def is_running(plugin_id):
    srv = _SERVERS.get(plugin_id)
    return bool(srv and srv.running())


def stop(plugin_id):
    """Arrete le serveur lance par Fooocus pour ce plugin (un serveur externe est
    seulement oublie). True si quelque chose tournait."""
    with _LOCK:
        srv = _SERVERS.pop(plugin_id, None)
    if srv is None:
        return False
    was_running = srv.running()
    _terminate(srv)
    return was_running


def stop_all():
    for pid in list(_SERVERS):
        try:
            stop(pid)
        except Exception:
            pass


atexit.register(stop_all)
