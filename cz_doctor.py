"""crispz-sdxl: a readable diagnosis for "it does not work".

czp.bat doctor [--remote URL] prints, in plain lines, what a person needs when
czp does not answer as expected: which branch and commit the app folder is on,
which Python and packages, what config.txt says, and above all whether a
running instance answers at the URL czp uses, as which tool, with which
endpoints. Every finding ends with the thing to do. Exit 0 when the instance
answers as crispz-sdxl, 1 otherwise.

Born on 2026-10-31: Mike ran "czp.bat caps" and got tool fooocus2026 with
route local, because no instance answered on 7865 and the local fallback
answered with the engine's identity. Nothing said so in a way a reader
could act on. This file is the answer to that.
"""
import importlib.util
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

import cz_core

EXPECTED_BRANCH = 'wip/crispz-sdxl'
BOOT_LINE = '[crispz] crispz-sdxl 0.1.0 endpoints bound'


def _git(*args):
    try:
        p = subprocess.run(['git', *args], cwd=cz_core.ROOT, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', timeout=10)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _module_version(name):
    if importlib.util.find_spec(name) is None:
        return None
    try:
        mod = __import__(name)
        return str(getattr(mod, '__version__', 'present'))
    except Exception as e:  # a broken install must not stop the report
        return f'broken ({type(e).__name__})'


def probe(url, timeout=4):
    """{'state': 'ok' | 'legacy' | 'foreign' | 'no_endpoint' | 'down' | 'error', ...}"""
    body = json.dumps({'data': ['{}']}).encode('utf-8')
    req = urllib.request.Request(f'{url.rstrip("/")}/run/cli_caps', data=body,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {'state': 'no_endpoint', 'detail': 'the server answers but has no /run/cli_caps'}
        return {'state': 'error', 'detail': f'HTTP {e.code}'}
    except urllib.error.URLError as e:
        return {'state': 'down', 'detail': str(e.reason)}
    except Exception as e:
        return {'state': 'error', 'detail': f'{type(e).__name__}: {e}'}
    out = ((payload or {}).get('data') or [None])[0]
    try:
        caps = json.loads(out) if isinstance(out, str) else out
    except ValueError:
        return {'state': 'error', 'detail': f'cli_caps did not answer JSON: {str(out)[:120]}'}
    if not isinstance(caps, dict):
        return {'state': 'error', 'detail': 'cli_caps answered something that is not an object'}
    tool = caps.get('tool')
    if tool == cz_core.TOOL:
        return {'state': 'ok', 'caps': caps}
    if tool == 'fooocus2026':
        return {'state': 'legacy', 'caps': caps}
    return {'state': 'foreign', 'caps': caps}


def report(url=None, out=None):
    """Writes the report, returns 0 when the instance answers as crispz-sdxl."""
    out = out or sys.stdout
    problems = []

    def line(text=''):
        out.write(text + '\n')

    line(f'crispz-sdxl doctor ({cz_core.TOOL} {cz_core.VERSION})')
    line(f'  app folder      : {cz_core.ROOT}')

    branch = _git('rev-parse', '--abbrev-ref', 'HEAD')
    head = _git('rev-parse', '--short', 'HEAD')
    if branch is None:
        line('  git             : not a git checkout, or git is missing (cannot tell which version runs)')
    else:
        line(f'  git             : branch {branch}, commit {head}')
        if branch != EXPECTED_BRANCH:
            problems.append(f'the app folder is on branch "{branch}", not {EXPECTED_BRANCH}: run '
                            f'"git checkout {EXPECTED_BRANCH}" in {cz_core.ROOT}, then restart the app')
    has_layer = os.path.isfile(os.path.join(cz_core.ROOT, 'cz_protocol.py'))
    line(f'  crispz layer    : {"present" if has_layer else "MISSING"} (cz_protocol.py)')
    if not has_layer:
        problems.append('cz_protocol.py is missing: this checkout predates crispz-sdxl')

    line(f'  python          : {sys.version.split()[0]} ({sys.executable})')
    for name, why in (('torch', 'the SDXL engine'), ('gradio', 'the app'), ('insightface', 'the faces op'),
                      ('onnxruntime', 'insightface')):
        v = _module_version(name)
        line(f'  {name:<16}: {v or "not installed"}  [{why}]')
        if name == 'insightface' and v is None:
            problems.append('insightface is not installed: cli_faces answers code 1 and caps.supports.faces '
                            'is false until "python -m pip install insightface" in the app\'s Python')
    if _module_version('torch'):
        try:
            import torch
            cuda = 'yes, ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no'
            line(f'  cuda            : {cuda}')
        except Exception as e:
            line(f'  cuda            : unknown ({e})')

    cfg = cz_core.load_config()
    line(f'  config.txt      : out_dir {cz_core.out_dir(cfg)}, tag {cfg["tag"]}, fake_engine {cfg["fake_engine"]}, '
         f'allow_model_switch {cfg["allow_model_switch"]}')
    if cz_core.fake_engine_enabled(cfg):
        line('  FAKE ENGINE     : on (CRISPZ_FAKE_ENGINE or crispz.fake_engine): placeholder images only')
    try:
        os.makedirs(cz_core.out_dir(cfg), exist_ok=True)
    except OSError as e:
        problems.append(f'the output folder cannot be created ({e})')

    import fooocus_protocol
    url = (url or fooocus_protocol.instance_url()).rstrip('/')
    line(f'  instance url    : {url}  (config cli_protocol.instance_url, env FOOOCUS_CLI_URL, --remote)')
    p = probe(url)
    state = p['state']
    if state == 'ok':
        caps = p['caps']
        eng = caps.get('engine') or {}
        line(f'  instance        : {caps.get("tool")} {caps.get("version")} on {eng.get("tool")} {eng.get("version")}'
             f'{" (FAKE engine)" if eng.get("fake") else ""}')
        line(f'  ops             : {", ".join(caps.get("ops", []))}')
        line(f'  models / loras  : {len(caps.get("models", []))} SDXL checkpoints, {len(caps.get("loras", []))} LoRAs'
             f'{", hidden " + json.dumps(caps["hidden"]) if caps.get("hidden") else ""}')
        line(f'  model loaded    : {caps.get("model_loaded")}, model switch by spec: {caps.get("model_switch")}')
        if not (caps.get('supports') or {}).get('faces'):
            line('  faces           : off on the instance (insightface missing there)')
    elif state == 'legacy':
        line(f'  instance        : Fooocus2026 {p["caps"].get("version")} with the custom-26 endpoints only')
        problems.append('the running app answers as fooocus2026: it was started before the crispz-sdxl branch '
                        'was checked out, or from another folder. Stop it, check the branch above, start run.bat '
                        f'again and wait for the console line "{BOOT_LINE}"')
    elif state == 'foreign':
        line(f'  instance        : another app answers ({p["caps"].get("tool")})')
        problems.append(f'port {url.rsplit(":", 1)[-1]} is used by another crispz app: start Fooocus on a free '
                        'port (--port N) and set cli_protocol.instance_url in config.txt')
    elif state == 'no_endpoint':
        line('  instance        : a web server answers but has no cli_caps endpoint')
        problems.append('something answers on this port without the protocol endpoints: an older Fooocus, or '
                        'another web app. Check the port and the branch of the running app')
    elif state == 'down':
        line(f'  instance        : nothing answers ({p["detail"]})')
        problems.append('no app is listening at this URL. Start run.bat (or boot_check_rtx5090.bat) in the app '
                        f'folder, wait for "{BOOT_LINE}" in its console (torch takes 30 to 60 s), then try '
                        'again. Started on another port: czp.bat doctor --remote http://127.0.0.1:PORT')
    else:
        line(f'  instance        : error ({p["detail"]})')
        problems.append('the instance answered something unexpected, see the line above')

    line()
    if problems:
        line(f'{len(problems)} thing(s) to fix:')
        for n, text in enumerate(problems, 1):
            line(f'  {n}. {text}')
        return 1
    line('All good: czp.bat caps / gen / edit / faces will run on this instance.')
    return 0


def main(url=None, out=None):
    try:
        return report(url, out)
    except Exception as e:  # the doctor itself must never die silently
        (out or sys.stdout).write(f'doctor failed: {type(e).__name__}: {e}\n')
        return 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != '--remote' else
                  (sys.argv[2] if len(sys.argv) > 2 else None)))
