"""crispz-sdxl: CLI protocol v1 of the crispz family, hidden Gradio endpoints.

The family drives an app through four hidden endpoints (POST /run/<name> on
Gradio 3.41, one JSON string in, one JSON string out):

  cli_caps   tool, version, supports {edit, faces, max_refs, ...}, models, loras
  cli_gen    text to image (op gen; upscale and inpaint ride the same endpoint)
  cli_edit   instruction edit of an input image (lot 2)
  cli_faces  face detection and embeddings of an image (lot 2)

A spec is {protocol: 1, op, prompt, input, refs, width, height, seed, steps,
guidance, model, out_dir, detail_faces, ...}; the answer carries the output
paths. Fooocus2026 already speaks this protocol as "fooocus2026" (custom-26,
fooocus_protocol.py, czp.bat): the parsing, the validation and the mapping
onto the Fooocus task are reused from there, not copied. This module is the
"crispz-sdxl" identity on top: the family tool name and version, the
capabilities computed from what cz_pipeline really runs, and the two extra
endpoints. Lot 3 (cz_ui.py) binds endpoints() to the Gradio app.

Everything here is importable without torch or gradio: caps must answer on a
machine without a GPU, and the tests run on CPU.
"""
import json

import cz_core
import cz_face
import cz_pipeline
import fooocus_protocol as fp

PROTOCOL = fp.PROTOCOL
TOOL = cz_core.TOOL
VERSION = cz_core.VERSION
OPS = ('caps', 'gen', 'edit', 'faces', 'upscale', 'inpaint')
ProtocolError = fp.ProtocolError

# 'local' when this module runs in its own process (a CLI), 'remote' once
# bind_gradio() wired it into the running app: there a spec's "model" is
# refused unless crispz.allow_model_switch is set (see cz_pipeline.generate).
ROUTE = 'local'


def parse_spec(spec_json):
    """A JSON string (or an already decoded dict) -> the spec dict and its op.
    ProtocolError code 2 when it is not an object, 3 for a foreign protocol."""
    if isinstance(spec_json, (bytes, bytearray)):
        spec_json = spec_json.decode('utf-8-sig')
    if isinstance(spec_json, str):
        try:
            spec = json.loads(spec_json or '{}')
        except ValueError as e:
            raise ProtocolError(f'the spec is not valid JSON: {e}', 2)
    else:
        spec = spec_json
    if spec is None:
        spec = {}
    if not isinstance(spec, dict):
        raise ProtocolError('the spec must be a JSON object', 2)
    if 'protocol' in spec and spec['protocol'] != PROTOCOL:
        raise ProtocolError(f'unsupported protocol {spec["protocol"]!r} (this tool speaks {PROTOCOL})', 3)
    return spec, str(spec.get('op') or 'gen')


def validate(spec, op):
    """Normalised spec and warnings, through the custom-26 validator (sizes to a
    multiple of 8, LoRA tags, refs capped, files checked on disk)."""
    return fp.validate_spec(spec, op)


def max_refs():
    """Image Prompt slots of Fooocus (config default_controlnet_image_count),
    read lazily: modules.config is heavy and writes config.txt on first import."""
    try:
        import modules.config as cfg
        return int(getattr(cfg, 'default_controlnet_image_count', fp.MAX_REFS))
    except Exception:
        return fp.MAX_REFS


def caps_dict(models=None, loras=None):
    """The cli_caps answer. supports.* is computed from cz_pipeline.IMPLEMENTED_OPS,
    so a client never reads a capability this lot does not run; planned ops are
    listed apart with their lot."""
    if models is None or loras is None:
        import modules.config as cfg
        models = list(cfg.model_filenames) if models is None else models
        loras = list(cfg.lora_filenames) if loras is None else loras
    ops = list(cz_pipeline.IMPLEMENTED_OPS)
    faces_ok = 'faces' in ops and cz_face.available()
    if 'faces' in ops and not faces_ok:
        ops.remove('faces')  # announced, not silent: the client reads supports.faces false
    fake = cz_core.fake_engine_enabled()
    cfg = cz_core.load_config()
    return {
        'ok': True, 'protocol': PROTOCOL, 'tool': TOOL, 'version': VERSION,
        'engine': {'tool': fp.TOOL, 'version': fp.tool_version(), 'fake': fake},
        'ops': ['caps'] + ops,
        'planned': dict(cz_pipeline.PLANNED_OPS),
        'route': ROUTE,
        'model_switch': ROUTE == 'local' or cfg['allow_model_switch'],
        'faces_engine': 'fake' if fake else ('insightface' if faces_ok else None),
        'edit_mode': 'img2img: Vary (Strong) with denoise, or inpaint of the mask; SDXL follows the '
                     'prompt as a description of the result, not as an instruction',
        'supports': {
            'edit': 'edit' in ops,
            'faces': faces_ok,
            'max_refs': min(max_refs(), fp.MAX_REFS),
            'loras': True, 'refs': 'gen' in ops, 'seed': True, 'negative': True,
            'arbitrary_size': True, 'inpaint': 'inpaint' in ops, 'img2img': 'upscale' in ops,
            'detail_faces': 'gen' in ops, 'detail_hands': 'gen' in ops,
        },
        'models': list(models), 'loras': list(loras),
        'model_loaded': cz_pipeline.model_loaded(),
        'out_dir': cz_pipeline.outputs_root(),
    }


def _answer(fn, spec_json):
    """Run fn(spec, op) and give ONE JSON string, errors included: a Gradio
    endpoint must never raise, the client reads {ok: false, code, error}."""
    try:
        spec, op = parse_spec(spec_json)
        res = fn(spec, op)
    except ProtocolError as e:
        res = {'ok': False, 'tool': TOOL, 'error': str(e), 'code': e.code}
    except Exception as e:  # a bug of ours must still answer in JSON, with its name
        res = {'ok': False, 'tool': TOOL, 'error': f'{type(e).__name__}: {e}', 'code': 1}
    return json.dumps(res, ensure_ascii=False)


def cli_caps(_spec_json=None):
    try:
        return json.dumps(caps_dict(), ensure_ascii=False)
    except Exception as e:
        return json.dumps({'ok': False, 'tool': TOOL, 'error': f'{type(e).__name__}: {e}', 'code': 1})


def cli_gen(spec_json):
    def run(spec, op):
        if op not in ('gen', 'upscale', 'inpaint'):
            raise ProtocolError(f'cli_gen runs gen, upscale and inpaint, not {op!r} '
                                '(edit: cli_edit, faces: cli_faces)', 3)
        return cz_pipeline.generate(spec, op, route=ROUTE)
    return _answer(run, spec_json)


def cli_edit(spec_json):
    def run(spec, op):
        if op not in ('gen', 'edit'):  # a spec without op, or op edit
            raise ProtocolError(f'cli_edit runs edit, not {op!r}', 3)
        return cz_pipeline.generate(spec, 'edit', route=ROUTE)
    return _answer(run, spec_json)


def cli_faces(spec_json):
    def run(spec, op):
        if op not in ('gen', 'faces'):
            raise ProtocolError(f'cli_faces runs faces, not {op!r}', 3)
        return cz_pipeline.generate(spec, 'faces', route=ROUTE)
    return _answer(run, spec_json)


def endpoints():
    """api_name -> handler, in the order they are bound."""
    return {'cli_caps': cli_caps, 'cli_gen': cli_gen, 'cli_edit': cli_edit, 'cli_faces': cli_faces}


def bind_gradio(gr):
    """Hidden buttons in the current gr.Blocks context, one per endpoint, out
    of the Gradio queue so POST /run/<api_name> answers (Gradio 3.41). Replaces
    the custom-26 pair in webui.py: cli_caps now answers as crispz-sdxl, with
    the engine named inside. Returns {api_name: button}."""
    global ROUTE
    ROUTE = 'remote'
    buttons = {}
    with gr.Row(visible=False):
        spec_in = gr.Textbox(visible=False)
        spec_out = gr.Textbox(visible=False)
        for name, handler in endpoints().items():
            button = gr.Button(visible=False)
            button.click(handler, inputs=spec_in, outputs=spec_out, api_name=name, queue=False,
                         show_progress=False)
            buttons[name] = button
    cz_core.log(f'{TOOL} {VERSION} endpoints bound: {", ".join(buttons)} (engine {fp.TOOL})')
    return buttons
