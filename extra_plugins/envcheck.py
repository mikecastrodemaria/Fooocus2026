"""custom-30 — Sanity check of a plugin venv before a run: the interrupted torch install.

The torch step is the longest of a plugin install (a 3 GB cu128 wheel). When Fooocus is
closed, restarted or killed in the middle of it, the venv is left with the small
dependencies in place and a `torch/` folder without its `torch-*.dist-info`: pip does
not list it, `import torch` dies on a DLL error (WinError 127 on cudnn_cnn64_9.dll on
Windows) and the plugin tab showed a raw traceback. Update never replays the torch step
(by design, custom-20), so nothing fixed it.

This module names the state and the fix, before the run (preflight) and after a failed
one (explain_failure). Standard library only, no Fooocus import.
"""
import glob
import os
import re

# A pip token that installs torch: "torch", "torch==2.7.1", "torch>=2", "torch<3"...
_TORCH_TOKEN_RE = re.compile(r'^torch(?:$|[=<>!~])')
# Windows: torch found the DLL but a dependency of it is wrong or missing.
_DLL_RE = re.compile(r'Error loading "[^"]*torch[\\/]lib[\\/][^"]+"|WinError 12[67]\b.*torch', re.IGNORECASE)
_NO_TORCH_RE = re.compile(r"No module named '?torch'?")


def site_packages(plugin_dir):
    """site-packages of the plugin venv (Windows or posix layout), or None."""
    venv = os.path.join(plugin_dir, '.venv')
    win = os.path.join(venv, 'Lib', 'site-packages')
    if os.path.isdir(win):
        return win
    for p in sorted(glob.glob(os.path.join(venv, 'lib', 'python3*', 'site-packages'))):
        return p
    return None


def uses_system_site_packages(plugin_dir):
    """True when pyvenv.cfg says include-system-site-packages = true (reuse_python)."""
    try:
        with open(os.path.join(plugin_dir, '.venv', 'pyvenv.cfg'), encoding='utf-8') as f:
            for line in f:
                k, _, v = line.partition('=')
                if k.strip() == 'include-system-site-packages':
                    return v.strip().lower() == 'true'
    except OSError:
        pass
    return False


def torch_state(plugin_dir):
    """'ok' | 'incomplete' | 'missing' | 'inherited' | 'no-venv'.

    'incomplete' = the torch folder is there but not its dist-info (pip metadata), the
    signature of an interrupted wheel install. 'inherited' = not in the venv, but the
    venv sees the base interpreter's packages (reuse_python), so it may come from there.
    """
    sp = site_packages(plugin_dir)
    if sp is None:
        return 'no-venv'
    pkg = os.path.isdir(os.path.join(sp, 'torch'))
    dist = [d for d in glob.glob(os.path.join(sp, 'torch-*.dist-info'))
            if os.path.isfile(os.path.join(d, 'RECORD'))]
    if pkg and dist:
        return 'ok'
    if pkg:
        return 'incomplete'
    if uses_system_site_packages(plugin_dir):
        return 'inherited'
    return 'missing'


def torch_step(manifest):
    """(strategy name, step) of the first install step whose cmd installs torch, else (None, None)."""
    strategies = ((manifest or {}).get('env') or {}).get('strategies') or {}
    for name, strat in strategies.items():
        for step in (strat or {}).get('steps', []) or []:
            cmd = step.get('cmd') or []
            if any(isinstance(t, str) and _TORCH_TOKEN_RE.match(t) for t in cmd):
                return name, step
    return None, None


def needs_torch(manifest):
    return torch_step(manifest)[1] is not None


def _quote(token):
    return f'"{token}"' if ' ' in token else token


def repair_command(manifest, plugin_dir):
    """The manifest's own torch step, rendered as a shell line the user can paste."""
    venv = os.path.join(plugin_dir, '.venv')
    if os.name == 'nt':
        ctx = {'venv_pip': os.path.join(venv, 'Scripts', 'pip.exe'),
               'venv_python': os.path.join(venv, 'Scripts', 'python.exe')}
    else:
        ctx = {'venv_pip': os.path.join(venv, 'bin', 'pip'),
               'venv_python': os.path.join(venv, 'bin', 'python')}
    ctx['plugin_dir'] = plugin_dir
    ctx['base_python'] = 'python'
    _, step = torch_step(manifest)
    tokens = list((step or {}).get('cmd') or [ctx['venv_pip'], 'install', 'torch'])
    out = []
    for t in tokens:
        t = str(t)
        for k, v in ctx.items():
            t = t.replace('{' + k + '}', v)
        out.append(_quote(t))
    return ' '.join(out)


def _fix_lines(manifest, plugin_dir):
    return ('Fix: open a terminal in ' + plugin_dir + ' and run\n  '
            + repair_command(manifest, plugin_dir)
            + '\n(or Manager > Install with Force, fresh_venv), then run again.')


def preflight(plugin_dir, manifest):
    """None when the venv looks usable, else the message for the Status box.

    Only plugins whose manifest installs torch are checked, and only for the cheap,
    unambiguous states: no venv, torch folder without pip metadata, no torch at all.
    """
    if not needs_torch(manifest):
        return None
    state = torch_state(plugin_dir)
    if state in ('ok', 'inherited'):
        return None
    if state == 'no-venv':
        return ('Plugin venv not found (.venv missing in ' + plugin_dir + '). '
                'Reinstall it: Manager > Install with Force.')
    if state == 'incomplete':
        head = ('torch is incomplete in the plugin venv: the torch folder is there but not '
                'its torch-*.dist-info, so its install was interrupted (window closed, '
                'Restart UI or lost download during the 3 GB wheel). pip does not list it '
                'and "import torch" fails on a DLL error.')
    else:
        head = 'torch is not installed in the plugin venv (the install step did not run to the end).'
    return head + '\n' + _fix_lines(manifest, plugin_dir)


def explain_failure(text, manifest=None, plugin_dir=None):
    """A one-paragraph hint for a failed run whose output shows a torch load problem, else ''."""
    text = text or ''
    if _NO_TORCH_RE.search(text):
        hint = 'Hint: torch is not importable in the plugin venv.'
    elif _DLL_RE.search(text):
        hint = ('Hint: a torch DLL failed to load. Usual causes: an interrupted torch install '
                '(torch folder without its dist-info), or a second CUDA/cuDNN copy earlier in '
                'the DLL search path (System32, a CUDA bin folder on PATH).')
    else:
        return ''
    if manifest is not None and plugin_dir:
        hint += '\n' + _fix_lines(manifest, plugin_dir)
    return hint
