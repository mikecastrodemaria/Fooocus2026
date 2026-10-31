"""crispz-sdxl: core of the crispz layer (configuration, paths, logging).

crispz-sdxl is a member of the crispz family (crispz-klein, crispz-flux3,
crispz-qwen-edit, crispz-krea2, crispz-studio) whose engine is SDXL as
Fooocus2026 implements it. The Fooocus engine stays untouched under modules/;
this file and its siblings (cz_protocol.py, cz_pipeline.py, cz_imageio.py)
are the family layer on top. See FORK.md for the plan and the lots.

This module must stay importable without torch, gradio or modules.config:
caps and the CPU tests read the configuration from here, and modules.config
costs about one second and writes config.txt on first import. The crispz
block is therefore read straight from config.txt with json, the same way
fooocus_protocol.instance_url() does for the cli_protocol block.
"""
import datetime
import json
import os
import sys

TOOL = 'crispz-sdxl'
VERSION = '0.1.0'
ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_KEY = 'crispz'

# Keys of the "crispz" block of config.txt (see config-sample.txt). The defaults
# follow the family conventions: outputs in out/YYYY-MM-DD/ next to the app,
# file names {date}_{tag}_seed{seed}_{w}x{h}. Fooocus itself keeps writing to
# its own path_outputs; which of the two is the final home of the images is an
# open question for lot 2 (FORK.md, "Open questions").
DEFAULTS = {
    'out_dir': 'out',
    'tag': 'sdxl',
    'port': 7865,
    'fake_engine': False,
    'keep_fooocus_ui': True,
    # The hidden endpoints of the running app keep the checkpoint the user
    # loaded (custom-26 rule: never swap the model under the user's feet, a
    # swap costs 10 to 30 s per task on the 5090 and the UI would reload its
    # own model right after). True lets a spec's "model" switch it anyway.
    'allow_model_switch': False,
    # Fooocus keeps writing its own file in path_outputs (log.html and the
    # Asset Browser read it); the crispz layer adds a copy in out/ with the
    # family name and the crispz chunk. False moves instead of copying.
    'keep_fooocus_outputs': True,
}


def log(message):
    """One console line, family style: a [crispz] prefix, never silent."""
    # sys.stdout is looked up at call time on purpose: fooocus_protocol.main()
    # swaps it for stderr while a CLI call runs, so the JSON line stays alone.
    print(f'[crispz] {message}', file=sys.stdout, flush=True)


def _read_config_file(path=None):
    path = path or os.environ.get('CRISPZ_CONFIG') or os.path.join(ROOT, 'config.txt')
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log(f'config.txt unreadable ({e}): the crispz defaults apply')
        return {}
    return data if isinstance(data, dict) else {}


def load_config(path=None):
    """The crispz block of config.txt, completed with the defaults. Unknown keys
    are kept (a later lot may read them) and announced once."""
    block = _read_config_file(path).get(CONFIG_KEY) or {}
    if not isinstance(block, dict):
        log(f'config.txt "{CONFIG_KEY}" is not an object: the defaults apply')
        block = {}
    cfg = dict(DEFAULTS)
    for key, value in block.items():
        if key not in DEFAULTS:
            log(f'config.txt "{CONFIG_KEY}.{key}" is not a known key (ignored)')
        cfg[key] = value
    cfg['fake_engine'] = bool(cfg['fake_engine'])
    for key in ('keep_fooocus_ui', 'allow_model_switch', 'keep_fooocus_outputs'):
        cfg[key] = bool(cfg[key])
    try:
        cfg['port'] = int(cfg['port'])
    except (TypeError, ValueError):
        log(f'config.txt "{CONFIG_KEY}.port" is not an integer: {DEFAULTS["port"]} applies')
        cfg['port'] = DEFAULTS['port']
    return cfg


def fake_engine_enabled(cfg=None):
    """CRISPZ_FAKE_ENGINE=1 (environment) or crispz.fake_engine (config.txt).

    The fake engine is a tiny PIL renderer: no torch, no model, no download. It
    exists so the protocol and the UI can be exercised on a CPU-only machine
    (the cloud sessions, the CI). It is never silent: cz_pipeline logs a line
    on every render it does."""
    env = os.environ.get('CRISPZ_FAKE_ENGINE')
    if env is not None:
        return env.strip() not in ('', '0', 'false', 'False', 'no')
    cfg = cfg or load_config()
    return bool(cfg['fake_engine'])


def out_dir(cfg=None):
    """Root of the family output tree (out/ by default), absolute. The
    CRISPZ_OUT_DIR environment variable wins, so the tests never write into
    the repository."""
    env = os.environ.get('CRISPZ_OUT_DIR')
    if env:
        return os.path.abspath(env)
    cfg = cfg or load_config()
    path = str(cfg['out_dir'])
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(ROOT, path))


def today():
    return datetime.date.today().strftime('%Y-%m-%d')
