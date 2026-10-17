"""custom-39 — Batch input: one Job Queue job per image of a folder (or of a file selection).

The Upscale or Variation tab gets a "Batch" accordion under the method radio: a folder
path on the machine running Fooocus (scanned here, hundreds of files are fine) or a
multi-file picker (upload from the browser, a handful of files). The list of files then
takes the place of the drop zone (custom-40), in that tab and in the Enhance tab, and
Generate or + Queue clones the current settings once per image, with the image put in the
slot of the active tab, and stacks the clones in the Job Queue (custom-14), which brings
order, pause, stop, restart survival and a Generate button that stays clickable.

Memory: a folder image is NOT decoded when the job is built. The job holds a small
reference {'__file__': path} that the queue runner resolves when the job starts
(`materialize`), so 200 photos in the queue cost nothing until their turn. Uploaded files
are decoded at once (they live in Gradio's temp dir, which does not survive a restart).

Outputs: with "Name outputs after the source file" the result of `photo_12.jpg` is
`photo_12_upscaled.png` (or `_vary`, `_enhanced`) in the usual dated output folder,
numbered `_2`, `_3` on collision, instead of the time stamp.

Standard library only at import; numpy and PIL are imported when an image is decoded.
No `modules.config` import: settings are read from it only when it is already loaded.
"""
import fnmatch
import os
import random
import re
import sys

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff')
FILE_KEY = '__file__'
SUPPORTED_TABS = {'uov': 'Upscale or Variation', 'enhance': 'Enhance'}
MIN_SEED = 0
MAX_SEED = 2 ** 63 - 1

_STEM_BAD_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]+')   # what Windows and Unix refuse; accents stay
_NAT_RE = re.compile(r'(\d+)')


# ------------------------------------------------------------------ settings

def _setting(key, default=None):
    """Reads modules.config.batch_input_setting without importing modules.config."""
    cfg = sys.modules.get('modules.config')
    getter = getattr(cfg, 'batch_input_setting', None) if cfg is not None else None
    if getter is None:
        return default
    try:
        value = getter(key, default)
        return default if value is None else value
    except Exception:
        return default


def allowed_roots():
    """Folders a batch may read from (config batch_input.allowed_roots). Empty = any."""
    raw = _setting('allowed_roots', [])
    if isinstance(raw, str):
        raw = [p for p in re.split(r'[;\n]', raw)]
    roots = []
    for p in raw or []:
        p = str(p).strip()
        if p:
            roots.append(os.path.realpath(os.path.expanduser(p)))
    return roots


def folder_allowed(path):
    """True when `path` is under one of the allowed roots (or when no root is set)."""
    roots = allowed_roots()
    if not roots:
        return True
    real = os.path.realpath(path)
    for root in roots:
        try:
            if os.path.commonpath([real, root]) == root:
                return True
        except ValueError:            # different drives on Windows
            continue
    return False


# ------------------------------------------------------------------- scanning

def natural_key(name):
    """'img2' < 'img10': digits compare as numbers, the rest case-insensitively."""
    return [int(t) if t.isdigit() else t.lower() for t in _NAT_RE.split(os.path.basename(name))]


def is_image_file(name):
    return os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS


def _match(name, pattern):
    pattern = (pattern or '').strip()
    if not pattern:
        return True
    pats = [p.strip() for p in pattern.split(',') if p.strip()]
    base = os.path.basename(name)
    return any(fnmatch.fnmatch(base.lower(), p.lower()) for p in pats)


def check_folder(path):
    """(ok, message, absolute path) for a folder typed by the user."""
    path = (path or '').strip().strip('"').strip("'")
    if not path:
        return False, 'Type the path of a folder on the machine running Fooocus.', ''
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(path):
        return False, f'Not a folder on this machine: {path}', path
    if not folder_allowed(path):
        return False, (f'Folder refused by batch_input.allowed_roots in config.txt: {path}. '
                       'Add it to the list, or empty the list to allow any folder.'), path
    return True, '', path


def scan_folder(path, recursive=False, pattern=''):
    """Image files of `path`, natural order; sub-folders only when `recursive`."""
    files = []
    if recursive:
        for root, dirs, names in os.walk(path):
            dirs.sort(key=natural_key)
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            files += [os.path.join(root, n) for n in sorted(names, key=natural_key)
                      if is_image_file(n) and _match(n, pattern)]
    else:
        files = [os.path.join(path, n) for n in sorted(os.listdir(path), key=natural_key)
                 if is_image_file(n) and _match(n, pattern) and
                 os.path.isfile(os.path.join(path, n))]
    return files


def scan_report(path, recursive=False, pattern=''):
    """Markdown line for the Scan button."""
    ok, msg, path = check_folder(path)
    if not ok:
        return msg
    files = scan_folder(path, recursive, pattern)
    return summarize(files, 'in ' + path)


def summarize(files, where=''):
    """One Markdown line; `files` holds paths or (path, eager) entries."""
    if not files:
        return f'No image found {where}.'.replace('  ', ' ')
    paths = [f[0] if isinstance(f, (tuple, list)) else f for f in files]
    names = [os.path.basename(p) for p in paths[:5]]
    more = f' (+{len(paths) - 5} more)' if len(paths) > 5 else ''
    return f'**{len(paths)} image(s)** {where}: {", ".join(names)}{more}'


def list_html(entries, note=''):
    """The block that takes the place of the drop zone while a batch is selected (custom-40)."""
    import html
    if not entries:
        return ''
    n = len(entries)
    names = [html.escape(os.path.basename(p)) for p, _ in entries[:30]]
    more = f'<li>... and {n - 30} more</li>' if n > 30 else ''
    note = f', {html.escape(note)}' if note else ''
    items = ''.join(f'<li>{name}</li>' for name in names)
    return ('<div style="border:1px dashed #888;border-radius:8px;padding:10px 14px;font-size:13px;">'
            f'<b>\U0001F4C1 Batch: {n} image(s)</b>, one job each{note}. Generate or + Queue stacks them.'
            f'<ol style="margin:6px 0 0 0;padding-left:22px;max-height:260px;overflow:auto;">{items}{more}</ol></div>')


def collect(folder, recursive, pattern, uploads):
    """Sources of a batch: [(path, eager)]. Folder files are lazy, uploads eager.
    Returns (entries, error message or '')."""
    entries = []
    uploads = uploads or []
    for f in uploads:
        p = getattr(f, 'name', f)
        if isinstance(p, dict):
            p = p.get('name') or p.get('path')
        if p and os.path.isfile(str(p)) and is_image_file(str(p)):
            entries.append((str(p), True))
    if (folder or '').strip():
        ok, msg, path = check_folder(folder)
        if not ok:
            return [], msg
        entries += [(p, False) for p in scan_folder(path, recursive, pattern)]
    if not entries:
        return [], 'Nothing to queue: scan a folder with images, or pick files.'
    return entries, ''


# ---------------------------------------------------------- lazy references

def file_ref(path):
    return {FILE_KEY: os.path.abspath(path)}


def is_file_ref(v):
    return isinstance(v, dict) and FILE_KEY in v and len(v) == 1


def load_image(path):
    """RGB uint8 HxWx3, like the Gradio Image component (EXIF orientation applied)."""
    import numpy as np
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.array(im.convert('RGB'))


def materialize(args):
    """Replace the file references of a job snapshot by the decoded images.
    Returns (args, '') or (args, reason) when a file is missing or unreadable:
    the caller skips the job (decision 3: skipped with a console line)."""
    out = list(args)
    for i, v in enumerate(out):
        if not is_file_ref(v):
            continue
        path = v[FILE_KEY]
        if not os.path.isfile(path):
            return out, f'source file missing: {path}'
        try:
            out[i] = load_image(path)
        except Exception as e:
            return out, f'source file unreadable: {path} ({e})'
    return out, ''


# ------------------------------------------------------------------ naming

def suffix_for(tab, uov_method=''):
    if tab == 'enhance':
        return 'enhanced'
    m = str(uov_method or '').lower()
    if 'vary' in m:
        return 'vary'
    return 'upscaled'


def output_stem(source_path, suffix):
    """'D:/x/photo 12.jpg', 'upscaled' -> 'photo 12_upscaled' (file-system safe)."""
    stem = os.path.splitext(os.path.basename(str(source_path)))[0]
    stem = _STEM_BAD_RE.sub('_', stem).strip(' ._') or 'image'
    return f'{stem}_{suffix}' if suffix else stem


def unique_output_path(folder, stem, extension):
    """folder/stem.ext, then stem_2.ext, stem_3.ext... (first free name)."""
    name = f'{stem}.{extension}'
    path = os.path.join(folder, name)
    i = 2
    while os.path.exists(path):
        name = f'{stem}_{i}.{extension}'
        path = os.path.join(folder, name)
        i += 1
    return path, name


def output_stem_of(meta):
    """The stem a queue job asks for, or None (jobs that are not from a batch)."""
    try:
        return (meta or {}).get('batch', {}).get('stem') or None
    except Exception:
        return None


# ------------------------------------------------------------------- jobs

def make_label(filename, tab, uov_method, base_model, seed):
    what = 'Enhance' if tab == 'enhance' else str(uov_method)
    base = str(base_model).rsplit('.', 1)[0]
    return f'{os.path.basename(filename)} | {what} | {base} | seed {seed}'


def build_jobs(base_args, entries, idx, keep_name=True, random_seed=True, rng=random):
    """One job per entry from the ctrls snapshot `base_args` (currentTask already popped).

    `idx` maps 'input_image_checkbox', 'current_tab', 'uov_method', 'uov_image',
    'enhance_image', 'enhance_checkbox', 'seed', 'base_model' to their index in base_args.
    Returns (jobs, message): jobs = [(args, label, meta)], message = '' or the reason why
    nothing was built (unsupported tab, Enhance tab without the Enhance box).
    """
    tab = str(base_args[idx['current_tab']])
    if tab not in SUPPORTED_TABS:
        names = ' or '.join(SUPPORTED_TABS.values())
        return [], f'Batch works with the {names} tab: select one of them first (current: {tab}).'
    if tab == 'enhance' and not base_args[idx['enhance_checkbox']]:
        return [], 'Enhance tab: tick the Enhance checkbox and enable at least one enhancement first.'
    if tab == 'uov' and str(base_args[idx['uov_method']]).lower() == 'disabled':
        return [], 'Upscale or Variation tab: pick a method (Vary or Upscale) first.'

    image_slot = idx['uov_image'] if tab == 'uov' else idx['enhance_image']
    uov_method = base_args[idx['uov_method']]
    suffix = suffix_for(tab, uov_method)
    seed = base_args[idx['seed']]
    jobs = []
    total = len(entries)
    for n, (path, eager) in enumerate(entries, 1):
        args = list(base_args)
        args[idx['input_image_checkbox']] = True
        if eager:
            try:
                args[image_slot] = load_image(path)
            except Exception as e:
                print(f'[Batch] skipped (unreadable): {path} ({e})')
                continue
        else:
            args[image_slot] = file_ref(path)
        if random_seed:
            seed = rng.randint(MIN_SEED, MAX_SEED)
        args[idx['seed']] = seed
        meta = {'batch': {'source': os.path.abspath(path), 'index': n, 'total': total,
                          'stem': output_stem(path, suffix) if keep_name else None}}
        jobs.append((args, make_label(path, tab, uov_method, args[idx['base_model']], seed), meta))
    if not jobs:
        return [], 'No readable image in the selection.'
    return jobs, ''
