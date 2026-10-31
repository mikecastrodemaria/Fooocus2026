"""crispz-sdxl: the family call convention over the Fooocus engine.

generate(spec, op) takes a CLI protocol v1 spec (a dict) and answers the
family result dict: {ok, tool, version, images, seed_used, ...}. Under the
hood gen, upscale and inpaint go through fooocus_protocol.run() (custom-26),
which maps the spec onto an AsyncTask snapshot (modules/task_args.py) and
queues it in the Fooocus worker thread (modules/async_worker.py): one GPU,
one queue, behind the user's own renders. edit is mapped here (see
build_edit_args) and executed the same way; faces goes to cz_face.py.

Two engines answer that worker contract:
  - the real one, modules.async_worker (torch, the SDXL checkpoint);
  - FakeWorker below when CRISPZ_FAKE_ENGINE=1 or crispz.fake_engine is set:
    a tiny PIL renderer that decodes the same snapshot by the positions
    recorded by task_args.build() and writes an image in the family layout.
    No torch, no model, no download: the protocol and the tests run on CPU.
    Its images are placeholders and every render says so on the console.

Outputs: Fooocus writes its own file in path_outputs (log.html and the Asset
Browser read it); the final image is then imported into out/YYYY-MM-DD/ with
the family name and the crispz chunk (cz_imageio.import_image), and copied
into the spec's out_dir when one is given. Lot 1 decision 5 of FORK.md.
"""
import os
import random
import re
import sys
import time

import numpy as np
from PIL import Image

import cz_core
import cz_face
import cz_imageio
import fooocus_protocol as fp

# What really executes. cz_protocol.caps_dict() reads it, so the capabilities
# a client sees never claim more than what runs. faces also needs a detector
# (cz_face.available()).
IMPLEMENTED_OPS = ('gen', 'edit', 'faces', 'upscale', 'inpaint')
PLANNED_OPS = {}
WORKER_OPS = ('gen', 'edit', 'upscale', 'inpaint')

# edit on SDXL: there is no instruction-following pipeline, so an edit is an
# image to image pass. Without a mask: Vary (Strong) on the input, the prompt
# describes the result, denoise = how far from the input (default 0.6, measured
# on Fooocus: 0.85 is the built in Strong, 0.5 keeps the composition and
# changes materials). Refs ride along as Image Prompt. With a mask: inpaint of
# the white area with the main prompt (Fooocus "Modify Content" semantics).
EDIT_DEFAULT_DENOISE = 0.6
EDIT_SPEC_FIELDS = fp.SPEC_FIELDS

ProtocolError = fp.ProtocolError

_SIZE = re.compile(r'(\d+)\D+(\d+)')


def _parse_size(aspect, fallback=(1024, 1024)):
    m = _SIZE.search(str(aspect or ''))
    return (int(m.group(1)), int(m.group(2))) if m else fallback


# ------------------------------------------------------------ fake engine

class FakeTask:
    """Stands in for modules.async_worker.AsyncTask: consumes the whole ctrl
    snapshot (fooocus_protocol.execute() checks that nothing is left over) and
    keeps the fields the renderer needs, read by name through the positions
    task_args.build() records, so no index is hard coded here either."""

    def __init__(self, args):
        import modules.task_args as task_args
        self.yields = []
        self.results = []
        self.last_stop = False
        self.fields = {}
        if not args:
            return
        _, idx = task_args.build(prompt='', seed=0)
        if len(args) != len(idx):
            raise ProtocolError(f'snapshot of {len(args)} ctrls, task_args.build() records {len(idx)}: '
                                'the AsyncTask order changed, update modules/task_args.py', 1)
        self.fields = {name: args[pos] for name, pos in idx.items()}
        del args[:]  # consumed, like the real AsyncTask.__init__ pops everything


class _ImmediateQueue(list):
    """The fake worker's async_tasks: a task is rendered the moment it is
    appended, in the caller's thread. fooocus_protocol.execute() then finds
    the 'finish' yield on its first poll."""

    def __init__(self, worker):
        super().__init__()
        self.worker = worker

    def append(self, task):
        super().append(task)
        self.worker.render(task)


class FakeWorker:
    AsyncTask = FakeTask

    def __init__(self):
        self.async_tasks = _ImmediateQueue(self)

    def render(self, task):
        from PIL import ImageDraw
        f = task.fields
        op = str(f.get('current_tab') or 'uov')
        seed = int(f.get('seed') or 0)
        width, height = _parse_size(f.get('aspect'))
        rng = (seed * 2654435761) & 0xFFFFFF
        colour = ((rng >> 16) & 255, (rng >> 8) & 255, rng & 255)

        if f.get('input_image_checkbox') and op == 'uov' and f.get('uov_input_image') is not None:
            src = Image.fromarray(f['uov_input_image'])
            method = str(f.get('uov_method'))
            factor = 1.5 if '1.5' in method else 2.0 if '2' in method else 1.0
            width, height = int(src.width * factor), int(src.height * factor)
            image = src.resize((width, height), Image.BILINEAR)
            if 'Vary' in method:  # a tint says how far from the input the fake went
                strength = f.get('overwrite_vary_strength')
                strength = 0.5 if strength is None or strength < 0 else float(strength)
                image = Image.blend(image, Image.new('RGB', image.size, colour), strength * 0.5)
            op = 'edit' if 'Strong' in method else 'upscale'
            label = f'FAKE {method}: {f.get("prompt")}'
        elif f.get('input_image_checkbox') and op == 'inpaint' and f.get('inpaint_input_image'):
            pair = f['inpaint_input_image']
            image = Image.fromarray(pair['image']).convert('RGB')
            mask = Image.fromarray(pair['mask'][:, :, 0]).convert('L')
            image.paste(Image.new('RGB', image.size, colour), mask=mask)
            width, height = image.size
            label = f'FAKE inpaint: {f.get("inpaint_additional_prompt") or f.get("prompt")}'
        else:
            op = 'gen'
            image = Image.new('RGB', (width, height), colour)
            draw = ImageDraw.Draw(image)
            for i in range(0, width, max(8, width // 8)):
                draw.line([(i, 0), (width - i, height)], fill=(255 - colour[0], 255 - colour[1], 255 - colour[2]))
            label = f'FAKE gen: {f.get("prompt")}'
        ImageDraw.Draw(image).text((4, 4), label[:200], fill=(255, 255, 255))

        loras = [f'{f[f"lora{i}_name"]}:{f[f"lora{i}_weight"]:g}' for i in range(8)
                 if f.get(f'lora{i}_enabled') and f.get(f'lora{i}_name') not in (None, 'None')]
        refs = sum(1 for i in range(8) if f.get(f'cn{i}_image') is not None)
        cfg = cz_core.load_config()
        meta = {'tool': cz_core.TOOL, 'version': cz_core.VERSION, 'engine': 'fake', 'op': op,
                'prompt': f.get('prompt'), 'negative': f.get('negative'), 'seed': seed,
                'width': width, 'height': height, 'steps': f.get('overwrite_step'),
                'guidance': f.get('cfg'), 'model': f.get('base_model'), 'loras': loras, 'refs': refs,
                'sampler': f.get('sampler'), 'scheduler': f.get('scheduler')}
        path = cz_imageio.output_path(cfg['tag'], seed, width, height)
        cz_imageio.save_image(image, path, meta, a1111=True)
        cz_core.log(f'FAKE engine rendered {op} {width}x{height} seed {seed} -> {path} '
                    '(placeholder image: no SDXL model was run)')
        task.results.append(path)
        task.yields.append(['finish', None])


_fake_worker = None


def worker():
    """The engine answering the worker contract: FakeWorker, or the real
    modules.async_worker. The real one needs torch; without it the error says
    how to get a CPU dry run instead of a bare ModuleNotFoundError."""
    global _fake_worker
    if cz_core.fake_engine_enabled():
        if _fake_worker is None:
            _fake_worker = FakeWorker()
            cz_core.log('FAKE engine active (CRISPZ_FAKE_ENGINE / crispz.fake_engine): '
                        'placeholder images, no SDXL model is loaded')
        return _fake_worker
    try:
        import modules.async_worker as real
    except ImportError as e:
        raise ProtocolError(f'the Fooocus engine is not available ({e}); set CRISPZ_FAKE_ENGINE=1 '
                            'for a CPU dry run with placeholder images', 1)
    return real


# ------------------------------------------------------------------- edit

def validate_edit_spec(spec):
    """edit = the gen fields (prompt, refs, loras, seed, steps, model) plus an
    input image, an optional mask and denoise. Returns (normalised, warnings)."""
    if not isinstance(spec, dict):
        raise ProtocolError('the spec must be a JSON object', 2)
    edit_only = ('input', 'mask', 'op', 'factor', 'detail_faces', 'detail_hands')
    gen_part = {k: v for k, v in spec.items() if k not in edit_only}
    s, warnings = fp.validate_spec(dict(gen_part, op='gen'), 'gen')
    s['op'] = 'edit'
    s['input'] = fp._existing_file(spec, 'input')
    s['mask'] = fp._existing_file(spec, 'mask') if spec.get('mask') else None
    if s['width'] is not None:
        warnings.append('width/height ignored for edit (the input size is kept)')
        s['width'] = s['height'] = None
    s['denoise'] = EDIT_DEFAULT_DENOISE if s['denoise'] is None else s['denoise']
    if s['mask']:
        with Image.open(s['mask']) as m:
            if m.convert('L').getextrema()[1] == 0:
                raise ProtocolError('the mask is all black: nothing to edit (white = edit)', 2)
    if spec.get('detail_faces') or spec.get('detail_hands'):
        warnings.append('detail_faces / detail_hands ignored for edit')
    return s, warnings


def build_edit_args(s, route, warnings):
    """Normalised edit spec -> AsyncTask snapshot. Reuses the gen mapping of
    custom-26 (LoRAs, refs, model, steps) then switches the input image tab."""
    import modules.flags as flags
    import modules.task_args as task_args
    args, info = fp.build_task_args(s, route, warnings)
    _, idx = task_args.build(prompt='', seed=0)
    image = Image.open(s['input']).convert('RGB')
    args[idx['input_image_checkbox']] = True
    if s['mask']:
        mask = Image.open(s['mask']).convert('L')
        if mask.size != image.size:
            warnings.append(f'mask {mask.size[0]}x{mask.size[1]} resized to the input {image.width}x{image.height}')
            mask = mask.resize(image.size, Image.NEAREST)
        m = np.array(mask)
        args[idx['current_tab']] = 'inpaint'
        args[idx['inpaint_input_image']] = {'image': np.array(image), 'mask': np.stack([m, m, m], axis=-1)}
        args[idx['inpaint_strength']] = s['denoise']
        if s['refs']:
            args[idx['mixing_image_prompt_and_inpaint']] = True
        info['mode'] = 'inpaint'
    else:
        args[idx['current_tab']] = 'uov'
        args[idx['uov_method']] = flags.strong_variation
        args[idx['uov_input_image']] = np.array(image)
        args[idx['overwrite_vary_strength']] = s['denoise']
        if s['refs']:
            args[idx['mixing_image_prompt_and_vary_upscale']] = True
        info['mode'] = 'img2img'
    return args, info


def run_edit(spec, route, engine, timeout):
    t0 = time.time()
    s, warnings = validate_edit_spec(spec)
    s['seed_used'] = s['seed'] if s['seed'] is not None else random.randint(0, 2 ** 32 - 1)
    args, info = build_edit_args(s, route, warnings)
    task = fp.execute(args, engine, timeout)
    images = [r for r in task.results if isinstance(r, str) and os.path.isfile(r)]
    if not images:
        raise ProtocolError('no image file was produced (generation interrupted?)', 1)
    return {'ok': True, 'protocol': fp.PROTOCOL, 'route': route, 'images': [os.path.abspath(images[-1])],
            'seed_used': s['seed_used'], 'refs_used': info['refs_used'], 'loras': info['loras'],
            'edit_mode': info['mode'], 'denoise': s['denoise'],
            'timings': {'total_s': round(time.time() - t0, 2)}, 'warnings': warnings}


# ------------------------------------------------------------------ faces

def run_faces(spec):
    """{input} -> {faces: [{box, mouth, score, embedding}], count, width, height}."""
    if not isinstance(spec, dict):
        raise ProtocolError('the spec must be a JSON object', 2)
    t0 = time.time()
    path = fp._existing_file(spec, 'input')
    if not cz_face.available():
        raise ProtocolError('faces: no detector (insightface is not installed: pip install insightface, '
                            'or CRISPZ_FAKE_ENGINE=1 for a CPU dry run)', 1)
    with Image.open(path) as im:
        width, height = im.size
    try:
        faces = cz_face.detect_faces_full(path)
    except RuntimeError as e:
        raise ProtocolError(str(e), 1)
    warnings = ['FAKE face detector: one centred face, hashed embedding'] if any(f.get('fake') for f in faces) else []
    for f in faces:
        f.pop('fake', None)
    return {'ok': True, 'protocol': fp.PROTOCOL, 'tool': cz_core.TOOL, 'version': cz_core.VERSION,
            'input': os.path.abspath(path), 'width': width, 'height': height, 'count': len(faces),
            'faces': faces, 'timings': {'total_s': round(time.time() - t0, 2)}, 'warnings': warnings}


# ------------------------------------------------------------------ entry

def _import_outputs(result, spec, op, fake):
    """The Fooocus files -> family files in out/ (fake ones already are), then a
    copy into the spec's out_dir when given."""
    cfg = cz_core.load_config()
    images = []
    for src in result['images']:
        if fake:
            images.append(src)
            continue
        meta = {'tool': cz_core.TOOL, 'version': cz_core.VERSION,
                'engine': f'{fp.TOOL} {fp.tool_version()}', 'op': op,
                'prompt': spec.get('prompt'), 'negative': spec.get('negative'),
                'seed': result['seed_used'], 'steps': spec.get('steps'), 'model': spec.get('model'),
                'loras': result.get('loras', []), 'refs': result.get('refs_used', 0),
                'denoise': result.get('denoise', spec.get('denoise'))}
        images.append(cz_imageio.import_image(src, meta, cfg['tag'], result['seed_used'],
                                              keep_source=cfg['keep_fooocus_outputs']))
    out_dir = spec.get('out_dir')
    if out_dir:
        out_dir = fp.caller_path(str(out_dir))
        images = [cz_imageio.copy_to(p, out_dir) for p in images]
    result['images'] = images
    return result


def generate(spec, op=None, route='local', timeout=3600):
    """Family entry point: a spec dict in, the result dict out. Raises
    ProtocolError (code 2 bad spec, 3 unsupported, 1 run error).

    route 'local': the spec may choose the checkpoint (CLI in its own process).
    route 'remote': the hidden endpoints of the running app; the spec's model
    is refused unless crispz.allow_model_switch is set (custom-26 rule)."""
    if not isinstance(spec, dict):
        raise ProtocolError('the spec must be a JSON object', 2)
    op = str(op or spec.get('op') or 'gen')
    if op in PLANNED_OPS:
        raise ProtocolError(f'{op} is not wired yet ({PLANNED_OPS[op]})', 3)
    if op not in IMPLEMENTED_OPS:
        raise ProtocolError(f'unknown op {op!r} (this tool runs {", ".join(IMPLEMENTED_OPS)})', 3)
    if op == 'faces':
        return run_faces(spec)

    engine = worker()
    fake = isinstance(engine, FakeWorker)
    if route == 'remote' and cz_core.load_config()['allow_model_switch']:
        route = 'local'  # fooocus_protocol refuses "model" on the remote route only
    # out_dir is handled here, after the family import, not by fooocus_protocol.run
    # (which would copy the Fooocus-named file): the spec it sees has none.
    inner = {k: v for k, v in spec.items() if k != 'out_dir'}
    if op == 'edit':
        result = run_edit(inner, route, engine, timeout)
    else:
        result = fp.run(inner, op, route, engine, timeout)
    result['tool'] = cz_core.TOOL
    result['version'] = cz_core.VERSION
    result['engine'] = {'tool': fp.TOOL, 'version': fp.tool_version(), 'fake': fake}
    if fake:
        result.setdefault('warnings', []).append('FAKE engine: placeholder image, no SDXL model was run')
    return _import_outputs(result, spec, op, fake)


def model_loaded():
    pipeline = sys.modules.get('modules.default_pipeline')
    base = getattr(pipeline, 'model_base', None) if pipeline else None
    return bool(base is not None and getattr(base, 'unet_with_lora', None) is not None)


def outputs_root():
    return cz_core.out_dir()


if __name__ == '__main__':
    print(f'{cz_core.TOOL} {cz_core.VERSION}, ops {IMPLEMENTED_OPS}, outputs in {outputs_root()}',
          file=sys.stderr)
    print(os.path.relpath(outputs_root(), cz_core.ROOT))
