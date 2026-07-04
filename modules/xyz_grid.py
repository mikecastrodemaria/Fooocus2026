"""custom-15 — Grille X/Y/Z (planches de comparaison bâties sur la Job Queue).

Principe : on fait varier 1 à 3 paramètres (X, Y, Z), chaque combo devient un
job dans la Job Queue (custom-14) avec un `meta` de groupe. Quand tous les jobs
d'un groupe sont finis, `on_job_done()` assemble la planche annotée (Pillow),
une par valeur de Z, dans outputs/xyz_grids/.

Les valeurs s'appliquent au snapshot ctrls par index. Les positions sont
calées sur l'ordre de consommation de AsyncTask.__init__ (modules/async_worker.py) :
  0 grid  1 prompt  2 neg  3 styles  4 perf  5 aspect  6 image_number
  7 format  8 seed  9 wildcards_order  10 sharpness  11 cfg  12 base_model
  13 refiner  14 refiner_switch  15..15+3n-1 loras (enabled, name, weight)
  puis base=15+3n : ... +17 sampler  +18 scheduler  +20 overwrite_step
Si l'ordre de ctrls change un jour, mettre à jour _indices() ET ce commentaire.
"""
import os
import threading
import time

# ---------------------------------------------------------------- axes

def _indices():
    import modules.config as _cfg
    n = int(getattr(_cfg, 'default_max_lora_number', 5))
    base = 15 + 3 * n
    return {
        'sharpness': 10,
        'cfg': 11,
        'checkpoint': 12,
        'lora1_weight': 17,          # 15 + 2 (enabled, name, weight du slot 1)
        'sampler': base + 17,
        'scheduler': base + 18,
        'steps': base + 20,          # overwrite_step
    }


# nom affiché -> (clé interne, parseur)
AXES = {
    'CFG': ('cfg', float),
    'Steps': ('steps', int),
    'Sampler': ('sampler', str),
    'Scheduler': ('scheduler', str),
    'Sharpness': ('sharpness', float),
    'Checkpoint': ('checkpoint', str),
    'LoRA 1 weight': ('lora1_weight', float),
    'Preset': ('preset', str),   # custom-15.1 : applique tout le preset sur la case
}
AXIS_CHOICES = ['(aucun)'] + list(AXES.keys())


def parse_values(axis_name, raw):
    """'4, 6, 8' -> [4.0, 6.0, 8.0] selon le type de l'axe. ValueError si vide/invalide."""
    if axis_name not in AXES:
        raise ValueError(f'axe inconnu: {axis_name}')
    _, caster = AXES[axis_name]
    vals = [v.strip() for v in str(raw).split(',') if v.strip()]
    if not vals:
        raise ValueError(f'{axis_name}: aucune valeur')
    return [caster(v) for v in vals]


def _load_preset(name):
    """Resout un nom de preset (partiel, insensible a la casse) et charge son JSON."""
    import modules.config as _cfg
    presets = [p for p in getattr(_cfg, 'available_presets', []) if p != 'initial']
    v0 = str(name).strip()
    exact = [p for p in presets if p.lower() == v0.lower()]
    cands = exact or [p for p in presets if v0.lower() in p.lower()]
    if not cands:
        raise ValueError(f'preset introuvable: "{v0}"')
    if len(cands) > 1:
        raise ValueError(f'preset ambigu "{v0}": {", ".join(cands[:4])}')
    return cands[0], (_cfg.try_get_preset_content(cands[0]) or {})


def _apply_preset(args, name):
    """Applique les champs generatifs du preset sur le snapshot ctrls.
    Le prompt et l'aspect ratio de l'utilisateur sont volontairement conserves
    (c'est le sujet de la comparaison qui doit rester constant)."""
    import modules.config as _cfg
    pname, p = _load_preset(name)
    idx = _indices()
    simple = {
        'default_styles': 3,
        'default_performance': 4,
        'default_sample_sharpness': 10,
        'default_cfg_scale': 11,
        'default_model': 12,
        'default_refiner': 13,
        'default_refiner_switch': 14,
        'default_sampler': idx['sampler'],
        'default_scheduler': idx['scheduler'],
        'default_overwrite_step': idx['steps'],
    }
    for key, i in simple.items():
        if key in p and p[key] is not None and i < len(args):
            args[i] = p[key]
    if p.get('default_prompt_negative'):
        args[2] = p['default_prompt_negative']
    loras = p.get('default_loras')
    if isinstance(loras, list):
        n = int(getattr(_cfg, 'default_max_lora_number', 5))
        flat = []
        for entry in loras[:n]:
            if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                flat += [bool(entry[0]), str(entry[1]), float(entry[2])]
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                flat += [True, str(entry[0]), float(entry[1])]
        while len(flat) < 3 * n:
            flat += [False, 'None', 1.0]
        args[15:15 + 3 * n] = flat
    return pname


def _resolve_checkpoint(v):
    """Tolerance de saisie : 'juggernaut' suffit si un seul modele correspond."""
    import modules.config as _cfg
    names = list(getattr(_cfg, 'model_filenames', []))
    v0 = str(v).strip()
    if v0 in names or not names:
        return v0
    low = v0.lower()
    cands = [n for n in names if low in n.lower()]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise ValueError(f'checkpoint introuvable: "{v0}" (voir dropdown Base Model)')
    raise ValueError(f'checkpoint ambigu "{v0}": {", ".join(cands[:4])}')


def _apply(args, axis_name, value):
    key = AXES[axis_name][0]
    if key == 'preset':
        _apply_preset(args, value)
        return
    if key == 'checkpoint':
        value = _resolve_checkpoint(value)
    idx = _indices()[key]
    if idx >= len(args):
        raise IndexError(f'{axis_name}: index {idx} hors limites ({len(args)} ctrls)')
    args[idx] = value


def _fmt(axis_name, value):
    if isinstance(value, float) and value == int(value):
        value = int(value)
    if axis_name == 'Checkpoint':
        value = os.path.splitext(os.path.basename(str(value)))[0]
    return f'{axis_name}={value}'


def expand(base_args, spec):
    """spec = [(axis_name|None, [values]), ...] pour X, Y, Z.
    Renvoie (jobs, group) où jobs = [(args, label, meta)] en ordre Z, Y, X."""
    axes = [(a, v) for a, v in spec if a and a in AXES]
    if not axes:
        raise ValueError('aucun axe sélectionné')
    xs = axes[0][1]
    ys = axes[1][1] if len(axes) > 1 else [None]
    zs = axes[2][1] if len(axes) > 2 else [None]
    x_name = axes[0][0]
    y_name = axes[1][0] if len(axes) > 1 else None
    z_name = axes[2][0] if len(axes) > 2 else None

    group_id = f'xyz_{int(time.time() * 1000)}'
    total = len(xs) * len(ys) * len(zs)
    jobs = []
    i = 0
    for zi, zv in enumerate(zs):
        for yi, yv in enumerate(ys):
            for xi, xv in enumerate(xs):
                args = list(base_args)
                args[6] = 1  # image_number force a 1 : une image par case
                _apply(args, x_name, xv)
                parts = [_fmt(x_name, xv)]
                if y_name is not None:
                    _apply(args, y_name, yv)
                    parts.append(_fmt(y_name, yv))
                if z_name is not None:
                    _apply(args, z_name, zv)
                    parts.append(_fmt(z_name, zv))
                i += 1
                label = f'[XYZ {i}/{total}] ' + ' | '.join(parts)
                meta = {'group': group_id, 'x': xi, 'y': yi, 'z': zi}
                jobs.append((args, label, meta))

    group = {
        'id': group_id,
        'expected': total,
        'cells': {},
        'x_labels': [_fmt(x_name, v) for v in xs],
        'y_labels': [_fmt(y_name, v) for v in ys] if y_name else [''],
        'z_labels': [_fmt(z_name, v) for v in zs] if z_name else [''],
        'nx': len(xs), 'ny': len(ys), 'nz': len(zs),
    }
    return jobs, group


# ------------------------------------------------------- suivi des groupes

_GROUPS = {}
_LOCK = threading.Lock()


def register_group(group):
    with _LOCK:
        _GROUPS[group['id']] = group


def on_job_done(meta, image_path):
    """Appelé par run_queue après chaque job. Renvoie la liste des planches
    assemblées si le groupe est complet, sinon None."""
    if not meta or 'group' not in meta:
        return None
    with _LOCK:
        g = _GROUPS.get(meta['group'])
        if g is None:
            return None
        if image_path:
            g['cells'][(meta['z'], meta['y'], meta['x'])] = image_path
        done = len(g['cells']) >= g['expected']
        if done:
            del _GROUPS[meta['group']]
    if not done:
        return None
    try:
        return assemble_group(g)
    except Exception as e:
        print(f'[XYZ] WARNING: assemblage impossible: {e}')
        return None


# ------------------------------------------------------------- assemblage

def _font(size):
    from PIL import ImageFont
    for name in ('arial.ttf', 'DejaVuSans.ttf', 'segoeui.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def assemble_group(g, cell_px=512):
    """Construit une planche PNG par valeur de Z. Renvoie la liste des chemins."""
    from PIL import Image, ImageDraw
    import modules.config as _cfg

    out_dir = os.path.join(_cfg.path_outputs, 'xyz_grids')
    os.makedirs(out_dir, exist_ok=True)

    # dimensions de cellule d'apres la premiere image disponible
    first = next(iter(g['cells'].values()))
    with Image.open(first) as im0:
        w0, h0 = im0.size
    cell_w = cell_px
    cell_h = max(1, round(cell_px * h0 / w0))

    font = _font(24)
    left = 200 if g['y_labels'] != [''] else 20
    top = 60
    pad = 4

    paths = []
    stamp = time.strftime('%Y%m%d_%H%M%S')
    for zi in range(g['nz']):
        W = left + g['nx'] * (cell_w + pad)
        H = top + g['ny'] * (cell_h + pad)
        canvas = Image.new('RGB', (W, H), (24, 24, 30))
        draw = ImageDraw.Draw(canvas)

        for xi, xl in enumerate(g['x_labels']):
            draw.text((left + xi * (cell_w + pad) + 10, 18), xl, fill=(230, 230, 235), font=font)
        if g['y_labels'] != ['']:
            for yi, yl in enumerate(g['y_labels']):
                draw.text((10, top + yi * (cell_h + pad) + 10), yl, fill=(230, 230, 235), font=font)
        if g['z_labels'] != ['']:
            draw.text((10, 2), g['z_labels'][zi], fill=(160, 200, 255), font=_font(18))

        for yi in range(g['ny']):
            for xi in range(g['nx']):
                p = g['cells'].get((zi, yi, xi))
                if not p or not os.path.isfile(p):
                    continue
                with Image.open(p) as im:
                    im = im.convert('RGB').resize((cell_w, cell_h), Image.LANCZOS)
                    canvas.paste(im, (left + xi * (cell_w + pad), top + yi * (cell_h + pad)))

        suffix = f'_z{zi + 1}' if g['nz'] > 1 else ''
        out = os.path.join(out_dir, f'xyz_{stamp}{suffix}.png')
        canvas.save(out)
        paths.append(out)
        print(f'[XYZ] Planche assemblee: {out}')
    return paths
