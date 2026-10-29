"""crispz-sdxl: the family call convention over the Fooocus engine.

generate(spec, op) takes a CLI protocol v1 spec (a dict) and answers the
family result dict: {ok, tool, version, images, seed_used, ...}. Under the
hood everything goes through fooocus_protocol.run() (custom-26), which maps
the spec onto an AsyncTask snapshot (modules/task_args.py) and queues it in
the Fooocus worker thread (modules/async_worker.py): one GPU, one queue,
behind the user's own renders.

Two engines answer that worker contract:
  - the real one, modules.async_worker (torch, the SDXL checkpoint);
  - FakeWorker below when CRISPZ_FAKE_ENGINE=1 or crispz.fake_engine is set:
    a tiny PIL renderer that decodes the same snapshot by the positions
    recorded by task_args.build() and writes an image in the family layout.
    No torch, no model, no download: the protocol and the tests run on CPU.
    Its images are placeholders and every render says so on the console.

Lot 1 wires gen, upscale and inpaint (the ops custom-26 already maps). edit
and faces are refused with code 3 until lot 2 (see FORK.md).
"""
import os
import re
import sys

import cz_core
import cz_imageio
import fooocus_protocol as fp

# What this lot really executes. cz_protocol.caps_dict() reads it, so the
# capabilities a client sees never claim more than what runs.
IMPLEMENTED_OPS = ('gen', 'upscale', 'inpaint')
PLANNED_OPS = {'edit': 'lot 2: Fooocus image prompt (IP-Adapter) or inpaint, Mike decides',
               'faces': 'lot 2: insightface detection and embeddings, as in the family cz_face.py'}

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
        from PIL import Image, ImageDraw
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
            label = f'FAKE {method}'
        elif f.get('input_image_checkbox') and op == 'inpaint' and f.get('inpaint_input_image'):
            pair = f['inpaint_input_image']
            image = Image.fromarray(pair['image']).convert('RGB')
            mask = Image.fromarray(pair['mask'][:, :, 0]).convert('L')
            image.paste(Image.new('RGB', image.size, colour), mask=mask)
            width, height = image.size
            label = f'FAKE inpaint: {f.get("inpaint_additional_prompt")}'
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


def generate(spec, op=None, timeout=3600):
    """Family entry point: a spec dict in, the result dict out. Raises
    ProtocolError (code 2 bad spec, 3 unsupported, 1 run error)."""
    if not isinstance(spec, dict):
        raise ProtocolError('the spec must be a JSON object', 2)
    op = str(op or spec.get('op') or 'gen')
    if op in PLANNED_OPS:
        raise ProtocolError(f'{op} is not wired yet ({PLANNED_OPS[op]})', 3)
    if op not in IMPLEMENTED_OPS:
        raise ProtocolError(f'unknown op {op!r} (this tool runs {", ".join(IMPLEMENTED_OPS)})', 3)
    engine = worker()
    fake = isinstance(engine, FakeWorker)
    # 'local' lets the spec choose the checkpoint; the remote route of custom-26
    # refuses it on purpose (never swap the model under the user of the UI). On
    # the hidden endpoints of the running app, cz_protocol passes route='remote'.
    result = fp.run(spec, op, 'fake' if fake else 'local', engine, timeout)
    result['tool'] = cz_core.TOOL
    result['version'] = cz_core.VERSION
    result['engine'] = {'tool': fp.TOOL, 'version': fp.tool_version(), 'fake': fake}
    if fake:
        result.setdefault('warnings', []).append('FAKE engine: placeholder image, no SDXL model was run')
    return result


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
