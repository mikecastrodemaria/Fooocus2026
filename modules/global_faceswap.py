"""custom-32 — Global FaceSwap: one face reference applied to every generation.

Fooocus's FaceSwap is an Image Prompt type (IP-Adapter face). It only applies from the
Image Prompt tab, or from Vary/Upscale and Inpaint when the two "Mixing Image Prompt
and ..." developer checkboxes are on, and never in plain text-to-image. This module
keeps one face reference in Settings and injects it into every task:

  - the face is saved next to config.txt (global_faceswap_face.png) and the settings
    in config.txt under "global_faceswap" (enabled, stop, weight);
  - AsyncTask calls inject() right after its Image Prompt tasks are built: a FaceSwap
    task is appended and the gates the worker checks are set for the current tab, so
    the face applies in text-to-image, Vary, Upscale, Inpaint and Enhance without ever
    turning on a Vary or an Inpaint the user did not ask for.

Identity is approximate (it is IP-Adapter face, applied during diffusion); the exact
face is the crispz-studio Face swap plugin. Reads modules.config only if it is already
loaded (import-safe, testable with a stub).
"""
import json
import os
import sys

import numpy as np
from PIL import Image

DEFAULTS = {'enabled': False, 'stop': 0.9, 'weight': 0.75}
CONFIG_KEY = 'global_faceswap'
FACE_FILENAME = 'global_faceswap_face.png'
FACESWAP_TYPE = 'FaceSwap'          # modules.flags.cn_ip_face


def _config():
    return sys.modules.get('modules.config')


def _config_path():
    cfg = _config()
    return os.path.abspath(getattr(cfg, 'config_path', './config.txt') if cfg else './config.txt')


def face_path():
    """Where the face reference lives: next to config.txt."""
    return os.path.join(os.path.dirname(_config_path()), FACE_FILENAME)


def settings():
    """Current settings (config.txt block over the defaults), clamped."""
    d = dict(DEFAULTS)
    cfg = _config()
    raw = getattr(cfg, 'global_faceswap_config', None) if cfg else None
    if isinstance(raw, dict):
        for k in DEFAULTS:
            if k in raw and raw[k] is not None:
                d[k] = raw[k]
    d['enabled'] = bool(d['enabled'])
    try:
        d['stop'] = min(1.0, max(0.0, float(d['stop'])))
    except (TypeError, ValueError):
        d['stop'] = DEFAULTS['stop']
    try:
        d['weight'] = min(2.0, max(0.0, float(d['weight'])))
    except (TypeError, ValueError):
        d['weight'] = DEFAULTS['weight']
    return d


def load_face():
    """The saved face as an RGB uint8 array, or None."""
    p = face_path()
    if not os.path.isfile(p):
        return None
    try:
        with Image.open(p) as im:
            return np.array(im.convert('RGB'))
    except Exception as e:
        print(f'[global-faceswap] could not read {p}: {e}')
        return None


def save_face(image):
    """Write the face (RGB array) next to config.txt; None removes it. Returns the path or None."""
    p = face_path()
    if image is None:
        if os.path.isfile(p):
            os.remove(p)
        return None
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    Image.fromarray(arr.astype(np.uint8)).save(p, format='PNG')
    return p


def save(enabled, image, stop, weight):
    """Persist everything (face file + config.txt block) and return the status HTML."""
    path = save_face(image)
    block = {'enabled': bool(enabled), 'stop': float(stop), 'weight': float(weight)}
    cfg_path = _config_path()
    try:
        current = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, 'r', encoding='utf-8') as f:
                current = json.load(f)
        current[CONFIG_KEY] = block
        with open(cfg_path, 'w', encoding='utf-8') as f:
            json.dump(current, f, indent=4, ensure_ascii=False)
    except Exception as e:
        return f'<span style="color:#e55;">Could not write config.txt: {e}</span>'
    cfg = _config()
    if cfg is not None:
        cfg.global_faceswap_config = dict(block)
    return status_html(block, path is not None)


def status_html(block=None, has_face=None):
    block = block or settings()
    if has_face is None:
        has_face = os.path.isfile(face_path())
    if block['enabled'] and has_face:
        return ('<span style="color:#4ecdc4;">Active: this face is applied to every generation '
                f'(stop {block["stop"]:.2f}, weight {block["weight"]:.2f}).</span>')
    if block['enabled']:
        return '<span style="color:#e5a12c;">Enabled but no face loaded: nothing is applied.</span>'
    return '<span style="color:#888;">Off.</span>'


def inject(task, faceswap_type=FACESWAP_TYPE, log=print):
    """Append the global face to `task.cn_tasks` and open the worker gates for its tab.

    Returns True when a task was added. Rules, so that nothing else changes:
      - Input Image off: turned on with the tab set to 'ip' (other Image Prompt tasks are
        dropped, the user had none in mind), the mixing flags are left alone;
      - tab 'ip': just append;
      - tab 'uov' / 'inpaint': the matching mixing flag is set; other tabs (enhance,
        describe, metadata): the Vary/Upscale one, which is enough to reach the 'cn'
        goal without adding a Vary or an Inpaint. When the user had no mixing flag on,
        the other Image Prompt tasks are dropped: they were not applied before either.
    """
    s = settings()
    if not s['enabled']:
        return False
    face = load_face()
    if face is None:
        log('[global-faceswap] enabled but no face saved: nothing applied')
        return False
    cn_tasks = getattr(task, 'cn_tasks', None)
    if not isinstance(cn_tasks, dict):
        return False
    tab = getattr(task, 'current_tab', 'uov')
    user_mixing = bool(getattr(task, 'mixing_image_prompt_and_vary_upscale', False)
                       or getattr(task, 'mixing_image_prompt_and_inpaint', False))

    def drop_others():
        for k in cn_tasks:
            if k != faceswap_type:
                cn_tasks[k] = []

    if not getattr(task, 'input_image_checkbox', False):
        task.input_image_checkbox = True
        task.current_tab = 'ip'
        drop_others()
    elif tab == 'ip':
        pass
    else:
        if not user_mixing:
            drop_others()
        if tab == 'inpaint':
            task.mixing_image_prompt_and_inpaint = True
        else:
            task.mixing_image_prompt_and_vary_upscale = True
    cn_tasks.setdefault(faceswap_type, []).append([face, float(s['stop']), float(s['weight'])])
    log(f'[global-faceswap] face applied (tab {tab}, stop {s["stop"]}, weight {s["weight"]})')
    return True
