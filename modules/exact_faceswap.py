"""custom-33 — Exact face swap on every output, through the crispz-studio Face swap plugin.

custom-32 guides the diffusion with the global face (IP-Adapter face, approximate). This
module pastes the *real* face on each final image, using the Face swap action of an
installed Extra plugin (crispz-studio: inswapper + occlusion mask + colour match +
CodeFormer). Same face reference as custom-32 (global_faceswap_face.png next to
config.txt).

Where it runs: modules.async_worker.save_and_log calls maybe_swap() on the final array,
before private_logger.log(), so metadata, the AI provenance and the watermark are applied
to the swapped image. One CLI call per image (the family server only serves /upscale), a
few seconds each. Off, unavailable (no plugin, no face, broken venv), no face detected or
any failure: the image is saved unchanged and the console says why, once per cause.

Reads modules.config only if it is already loaded; the Extra subsystem is imported lazily.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from modules import global_faceswap

DEFAULTS = {'enabled': False, 'keep_original': False, 'offload_host': False, 'timeout': 600}
CONFIG_KEY = 'exact_faceswap'
FACESWAP_FLAG = '--faceswap-only'
INSTALL_ROOT_OVERRIDE = None        # tests point it at a temporary plugins folder

_warned = {'msg': None}


def _config():
    return sys.modules.get('modules.config')


def _extra():
    try:
        import extra_plugins
        return extra_plugins
    except Exception:
        return None


def settings():
    d = dict(DEFAULTS)
    cfg = _config()
    raw = getattr(cfg, 'exact_faceswap_config', None) if cfg else None
    if isinstance(raw, dict):
        for k in DEFAULTS:
            if k in raw and raw[k] is not None:
                d[k] = raw[k]
    for k in ('enabled', 'keep_original', 'offload_host'):
        d[k] = bool(d[k])
    try:
        d['timeout'] = int(min(3600, max(30, int(d['timeout']))))
    except (TypeError, ValueError):
        d['timeout'] = DEFAULTS['timeout']
    return d


def save(enabled, keep_original, offload_host):
    """Persist the block in config.txt and return the status HTML."""
    block = {'enabled': bool(enabled), 'keep_original': bool(keep_original),
             'offload_host': bool(offload_host), 'timeout': settings()['timeout']}
    err = global_faceswap.write_config_block(CONFIG_KEY, block)
    if err:
        return f'<span style="color:#e55;">Could not write config.txt: {err}</span>'
    cfg = _config()
    if cfg is not None:
        cfg.exact_faceswap_config = dict(block)
    _warned['msg'] = None
    return status_html(block)


def find_action():
    """(plugin, action) of the first installed plugin with a Face swap action, else None."""
    ext = _extra()
    if ext is None:
        return None
    root = INSTALL_ROOT_OVERRIDE or ext.INSTALL_ROOT
    for plugin in ext.registry.list_plugins(root):
        for action in ext.manifest.actions(plugin['manifest']):
            if (FACESWAP_FLAG in action['args'] or action['id'] == 'faceswap') and action['image_params']:
                return plugin, action
    return None


def availability():
    """(ok, message): can a swap run right now?"""
    ext = _extra()
    if ext is None:
        return False, 'the Extra plugins subsystem is not available'
    found = find_action()
    if found is None:
        return False, ('no installed plugin offers a Face swap action: install crispz-studio from '
                       'Extra > Manager (https://github.com/mikecastrodemaria/crispz-studio)')
    plugin, action = found
    if not os.path.isfile(global_faceswap.face_path()):
        return False, 'no face reference loaded above'
    try:
        from extra_plugins import envcheck
        problem = envcheck.preflight(plugin['dir'], plugin['manifest'])
    except Exception:
        problem = None
    if problem:
        return False, f'{plugin["name"]}: ' + problem.splitlines()[0]
    return True, f'ready: {plugin["name"]} / {action["label"]}'


def status_html(block=None):
    block = block or settings()
    if not block['enabled']:
        return '<span style="color:#888;">Exact swap off.</span>'
    ok, msg = availability()
    if ok:
        extras = []
        if block['keep_original']:
            extras.append('original kept')
        if block['offload_host']:
            extras.append('VRAM freed before each swap')
        tail = f' ({", ".join(extras)})' if extras else ''
        return f'<span style="color:#4ecdc4;">Exact swap active, {msg}{tail}.</span>'
    return f'<span style="color:#e5a12c;">Exact swap enabled but not possible: {msg}.</span>'


def swap(img, log=print):
    """Run the plugin on `img` (RGB array). Returns the swapped array, or None (kept as is)."""
    found = find_action()
    if found is None:
        return None
    plugin, action = found
    ext = _extra()
    face = global_faceswap.face_path()
    tmp = tempfile.mkdtemp(prefix='exact-faceswap-')
    try:
        in_path = os.path.join(tmp, 'input.png')
        Image.fromarray(np.asarray(img).astype(np.uint8)).save(in_path)
        out_dir = os.path.join(tmp, 'out')
        os.makedirs(out_dir, exist_ok=True)
        image_args = [(ip['arg'], face) for ip in action['image_params']]
        cmd = ext.runner.build_upscale_command(
            plugin['manifest'], plugin['dir'], input_path=in_path, output_dir=out_dir,
            param_values={}, esrgan_dir=None, report_vram=False,
            extra_args=action['args'], image_args=image_args)
        if settings()['offload_host']:
            ext.offload_host_models()
        t0 = time.time()
        res = ext.runner.run_upscale(cmd, plugin['dir'], timeout=settings()['timeout'])
        if not res['ok']:
            tail = (res.get('stderr') or '').strip().splitlines()[-3:]
            log(f'[exact-faceswap] {plugin["id"]} failed (code {res["returncode"]}), image kept as is: '
                + ' | '.join(tail))
            return None
        with Image.open(res['outputs'][-1]) as im:
            out = np.array(im.convert('RGB'))
        log(f'[exact-faceswap] face pasted by {plugin["id"]} in {time.time() - t0:.1f}s')
        return out
    except Exception as e:
        log(f'[exact-faceswap] error, image kept as is: {e}')
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def maybe_swap(img, log=print):
    """(image to save, swapped). Off, unavailable or failed: the image unchanged."""
    if not settings()['enabled']:
        return img, False
    ok, msg = availability()
    if not ok:
        if _warned['msg'] != msg:
            log('[exact-faceswap] skipped: ' + msg)
            _warned['msg'] = msg
        return img, False
    out = swap(img, log=log)
    if out is None:
        return img, False
    return out, True
