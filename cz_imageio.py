"""crispz-sdxl: family output layout and image metadata.

Layout: <out_dir>/YYYY-MM-DD/{date}_{tag}_seed{seed}_{w}x{h}{index}.png, with
the generation parameters in a "crispz" PNG text chunk (JSON) and in a .json
sidecar next to the image, like the other crispz apps. An A1111 "parameters"
chunk is written too when asked, so the image loads in the usual viewers.

Lot 1 only needs enough of this for the fake engine; the real Fooocus
outputs (modules/private_logger.py, log.html, fooocus / a1111 metadata
schemes) are mapped onto this layout in lot 2.
"""
import json
import os

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import cz_core

CHUNK = 'crispz'


def day_dir(base=None, date=None):
    path = os.path.join(base or cz_core.out_dir(), date or cz_core.today())
    os.makedirs(path, exist_ok=True)
    return path


def output_path(tag, seed, width, height, base=None, date=None, ext='png'):
    """Next free path for an image, family naming. A collision (same seed and
    size, the same day) gets _2, _3... so nothing is ever overwritten."""
    folder = day_dir(base, date)
    stem = f'{date or cz_core.today()}_{tag}_seed{seed}_{width}x{height}'
    path = os.path.join(folder, f'{stem}.{ext}')
    index = 2
    while os.path.exists(path):
        path = os.path.join(folder, f'{stem}_{index}.{ext}')
        index += 1
    return path


def a1111_parameters(meta):
    """The A1111 "parameters" text, from the family metadata."""
    lines = [str(meta.get('prompt') or '')]
    if meta.get('negative'):
        lines.append(f'Negative prompt: {meta["negative"]}')
    parts = []
    for label, key in (('Steps', 'steps'), ('Seed', 'seed'), ('CFG scale', 'guidance'), ('Model', 'model')):
        if meta.get(key) not in (None, ''):
            parts.append(f'{label}: {meta[key]}')
    if meta.get('width') and meta.get('height'):
        parts.append(f'Size: {meta["width"]}x{meta["height"]}')
    lines.append(', '.join(parts))
    return '\n'.join(lines)


def save_image(image, path, meta, a1111=False):
    """PNG with the crispz chunk (+ A1111 parameters on demand) and the .json
    sidecar. Returns the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    info = PngInfo()
    info.add_text(CHUNK, json.dumps(meta, ensure_ascii=False))
    if a1111:
        info.add_text('parameters', a1111_parameters(meta))
    image.save(path, pnginfo=info)
    with open(os.path.splitext(path)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def read_meta(path):
    """The crispz metadata of an image, from the chunk, else the sidecar, else {}."""
    try:
        with Image.open(path) as im:
            raw = (im.text if hasattr(im, 'text') else im.info).get(CHUNK)
        if raw:
            return json.loads(raw)
    except (OSError, ValueError):
        pass
    sidecar = os.path.splitext(path)[0] + '.json'
    if os.path.isfile(sidecar):
        with open(sidecar, encoding='utf-8') as f:
            return json.load(f)
    return {}


def import_image(src, meta, tag, seed, base=None, keep_source=True):
    """A Fooocus output (modules/private_logger.py naming) -> a family file in
    out/YYYY-MM-DD/ with the crispz chunk and the sidecar. The A1111
    "parameters" chunk Fooocus wrote (scheme a1111) is kept as it is; with the
    fooocus scheme, one is written from the family metadata. The source file
    stays by default: log.html and the Asset Browser point at it."""
    with Image.open(src) as im:
        image = im.convert('RGB') if im.mode not in ('RGB', 'RGBA') else im.copy()
        text = dict(im.text) if hasattr(im, 'text') else {}
    width, height = image.size
    meta = dict(meta, width=width, height=height, source=os.path.abspath(src))
    path = output_path(tag, seed, width, height, base)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    info = PngInfo()
    info.add_text(CHUNK, json.dumps(meta, ensure_ascii=False))
    for key in ('parameters', 'fooocus_scheme'):
        if text.get(key):
            info.add_text(key, text[key])
    if not text.get('parameters'):
        info.add_text('parameters', a1111_parameters(meta))
    image.save(path, pnginfo=info)
    with open(os.path.splitext(path)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    if not keep_source:
        try:
            os.remove(src)
        except OSError as e:
            cz_core.log(f'could not remove the Fooocus file {src} ({e}): both copies stay')
    return path


def copy_to(path, out_dir):
    """The image and its sidecar copied into out_dir (a dataset folder of the
    caller); returns the new image path."""
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.basename(path))
    if os.path.abspath(dst) == os.path.abspath(path):
        return path
    shutil.copy2(path, dst)
    sidecar = os.path.splitext(path)[0] + '.json'
    if os.path.isfile(sidecar):
        shutil.copy2(sidecar, os.path.splitext(dst)[0] + '.json')
    return dst
