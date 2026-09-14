"""custom-26 — Protocole CLI de la famille crispz (v1) pour Fooocus2026.

Contrat : comics2crispz/docs/CLI_PROTOCOL.md. Une spec JSON en entree, UNE ligne JSON
en sortie, codes 0 ok / 1 erreur d'execution / 2 spec invalide / 3 op ou protocole non
supporte / 4 pas de route. Fooocus2026 devient un moteur SDXL de la famille :
comics2crispz, un script de nuit ou un agent le pilotent comme crispz-studio.

  czp.bat caps
  czp.bat gen     --spec spec.json      (--spec - : spec sur stdin)
  czp.bat upscale --spec spec.json
  czp.bat inpaint --spec spec.json
  options : --remote URL (force cette instance), --local (force ce process),
            --timeout S (defaut 3600)

Routage (protocole §6) : par defaut on sonde l'instance Fooocus qui tourne
(config.txt cli_protocol.instance_url, sinon http://127.0.0.1:7865) par son endpoint
cache cli_caps ; l'instance execute dans son thread worker, derriere les rendus de
l'utilisateur (une GPU, une file). Sans instance : execution dans ce process, qui
charge le backend comme xyz_cli.py. L'identite de l'outil qui repond est verifiee :
une autre app sur le port n'est jamais prise pour Fooocus.

Correspondance avec Fooocus :
  gen      txt2img SDXL. width/height -> resolution exacte ; steps -> overwrite_step ;
           loras (+ balises <lora:...> du prompt) -> slots LoRA ; refs -> Image Prompt
           (IP-Adapter), 4 au plus ; detail_faces / detail_hands -> onglets Enhance
           (masque 'face' / 'hand', mode Improve Detail).
  upscale  factor 1 = Vary (force = denoise) ; 1.5 / 2 = Upscale Fooocus (force =
           denoise ; denoise 0 en 2x = Upscale Fast). Autre facteur : exit 2.
  inpaint  masque blanc = a redessiner ; denoise = force ; prompt LOCAL.
  edit     exit 3 : Fooocus n'a pas de pipeline d'edition par instruction.
"""
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
PROTOCOL = 1
TOOL = 'fooocus2026'
OPS = ('caps', 'gen', 'upscale', 'inpaint')
MAX_REFS = 4
UPSCALE_FACTORS = (1.0, 1.5, 2.0)
SPEC_FIELDS = ('protocol', 'op', 'prompt', 'negative', 'width', 'height', 'seed', 'steps',
               'refs', 'loras', 'model', 'out_dir', 'count', 'input', 'factor', 'denoise',
               'mask', 'detail_faces', 'detail_hands')
_LORA_TAG = re.compile(r'<lora:([^:>]+?)(?::([0-9.]+))?>', re.IGNORECASE)


class ProtocolError(Exception):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


def tool_version():
    try:
        import fooocus_version
        return str(fooocus_version.version)
    except Exception:
        return '?'


def caller_path(path):
    """Chemin relatif = relatif au dossier de l'appelant (czp.bat fait un cd)."""
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(os.environ.get('CZP_CALLER_CWD') or os.getcwd(), path))


# ------------------------------------------------------------------- spec

def _int(spec, key, lo=None, hi=None):
    v = spec.get(key)
    if v is None:
        return None
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ProtocolError(f'"{key}" must be an integer (got {v!r})', 2)
    if (lo is not None and n < lo) or (hi is not None and n > hi):
        raise ProtocolError(f'"{key}" out of range [{lo}, {hi}] (got {n})', 2)
    return n


def _existing_file(spec, key):
    p = spec.get(key)
    if not p:
        raise ProtocolError(f'"{key}" is required for this op', 2)
    p = caller_path(str(p))
    if not os.path.isfile(p):
        raise ProtocolError(f'"{key}" not found on disk: {p}', 2)
    return p


def validate_spec(spec, op):
    """Controle + normalisation, sans charger Fooocus. Renvoie (spec normalisee, warnings).
    Leve ProtocolError(code 2 ou 3)."""
    if not isinstance(spec, dict):
        raise ProtocolError('the spec must be a JSON object', 2)
    if spec.get('protocol') != PROTOCOL:
        raise ProtocolError(f'unsupported protocol {spec.get("protocol")!r} (this tool speaks {PROTOCOL})', 3)
    if op == 'edit':
        raise ProtocolError('edit is not supported: Fooocus has no instruction-edit pipeline '
                            '(caps.supports.edit is false)', 3)
    if op not in ('gen', 'upscale', 'inpaint'):
        raise ProtocolError(f'unknown op {op!r}', 3)
    warnings = []
    if spec.get('op') not in (None, op):
        warnings.append(f'spec op "{spec.get("op")}" ignored: the command is "{op}"')
    for key in spec:
        if key not in SPEC_FIELDS:
            warnings.append(f'unknown field "{key}" ignored')
    if spec.get('count') not in (None, 1):
        warnings.append(f'count {spec.get("count")} forced to 1 (one image per call, the caller loops)')

    s = {'op': op}
    raw_prompt = str(spec.get('prompt') or '')
    tagged = [(name.strip(), float(w) if w else None) for name, w in _LORA_TAG.findall(raw_prompt)]
    # balises retirees, puis ni double espace ni espace orpheline avant la ponctuation
    prompt = _LORA_TAG.sub('', raw_prompt)
    prompt = re.sub(r'\s+([,.;:])', r'\1', re.sub(r'\s{2,}', ' ', prompt)).strip(' ,')
    if tagged and not prompt:
        raise ProtocolError('the prompt contains only <lora:...> tags: describe the image too', 2)
    s['prompt'] = prompt
    explicit = []
    for entry in spec.get('loras') or []:
        name, sep, weight = str(entry).rpartition(':')
        if not sep:
            name, weight = str(entry), ''
        try:
            w = float(weight) if weight != '' else None
        except ValueError:
            raise ProtocolError(f'invalid LoRA weight in "{entry}" (expected name.safetensors:0.8)', 2)
        explicit.append((name.strip(), w))
    loras, seen = [], set()
    for name, w in explicit + tagged:  # une entree explicite gagne sur une balise du prompt
        if name and name.lower() not in seen:
            seen.add(name.lower())
            loras.append((name, w))
    s['loras'] = loras
    s['negative'] = None if spec.get('negative') is None else str(spec.get('negative'))
    seed = spec.get('seed')
    if seed in (None, -1, '-1'):
        s['seed'] = None
    else:
        s['seed'] = _int(spec, 'seed', 0, 2 ** 63 - 1)
    s['steps'] = _int(spec, 'steps', 1, 200)
    s['model'] = str(spec['model']) if spec.get('model') else None
    s['out_dir'] = caller_path(str(spec['out_dir'])) if spec.get('out_dir') else None

    width, height = _int(spec, 'width', 256, 4096), _int(spec, 'height', 256, 4096)
    if (width is None) != (height is None):
        raise ProtocolError('"width" and "height" go together', 2)
    if width is not None:
        w8, h8 = width - width % 8, height - height % 8
        if (w8, h8) != (width, height):
            warnings.append(f'size {width}x{height} rounded down to {w8}x{h8} (multiple of 8)')
        if op != 'gen':
            warnings.append(f'width/height ignored for {op} (the input size is kept)')
        width, height = w8, h8
    s['width'], s['height'] = width, height

    refs = []
    for r in spec.get('refs') or []:
        p = caller_path(str(r))
        if not os.path.isfile(p):
            raise ProtocolError(f'ref not found on disk: {p}', 2)
        refs.append(p)
    if len(refs) > MAX_REFS:
        warnings.append(f'{len(refs)} refs given, only the first {MAX_REFS} are used')
        refs = refs[:MAX_REFS]
    if refs and op != 'gen':
        warnings.append(f'refs ignored for {op}')
        refs = []
    s['refs'] = refs

    s['detail_faces'] = bool(spec.get('detail_faces'))
    s['detail_hands'] = bool(spec.get('detail_hands'))
    if op != 'gen' and (s['detail_faces'] or s['detail_hands']):
        warnings.append(f'detail_faces / detail_hands ignored for {op}')
        s['detail_faces'] = s['detail_hands'] = False

    denoise = spec.get('denoise')
    if denoise is not None:
        try:
            denoise = float(denoise)
        except (TypeError, ValueError):
            raise ProtocolError(f'"denoise" must be a number (got {spec.get("denoise")!r})', 2)
        if not 0.0 <= denoise <= 1.0:
            raise ProtocolError(f'"denoise" out of range [0, 1] (got {denoise})', 2)
    s['denoise'] = denoise

    if op == 'gen':
        if not prompt:
            raise ProtocolError('"prompt" is required for gen', 2)
    elif op == 'upscale':
        s['input'] = _existing_file(spec, 'input')
        factor = spec.get('factor', 2.0)
        try:
            factor = float(factor)
        except (TypeError, ValueError):
            raise ProtocolError(f'"factor" must be a number (got {spec.get("factor")!r})', 2)
        if factor not in UPSCALE_FACTORS:
            raise ProtocolError(f'factor {factor} is not supported by Fooocus: use 1 (variation), '
                                f'1.5 or 2', 2)
        s['factor'] = factor
    elif op == 'inpaint':
        s['input'] = _existing_file(spec, 'input')
        s['mask'] = _existing_file(spec, 'mask')
        from PIL import Image
        with Image.open(s['mask']) as m:
            if m.convert('L').getextrema()[1] == 0:
                raise ProtocolError('the mask is all black: nothing to redraw (white = redraw)', 2)
            with Image.open(s['input']) as im:
                if m.size != im.size:
                    warnings.append(f'mask {m.size[0]}x{m.size[1]} resized to the input '
                                    f'{im.size[0]}x{im.size[1]}')
    return s, warnings


# -------------------------------------------------------- snapshot Fooocus

def _resolve_model(name, filenames):
    if name in filenames:
        return name
    low = name.lower()
    cands = [f for f in filenames if low in f.lower()]
    if len(cands) == 1:
        return cands[0]
    if not cands:
        raise ProtocolError(f'checkpoint not found: "{name}" (see caps.models)', 2)
    raise ProtocolError(f'checkpoint "{name}" is ambiguous: {", ".join(cands[:4])}', 2)


def build_task_args(s, route, warnings):
    """Spec normalisee -> snapshot ctrls AsyncTask (+ infos pour la reponse)."""
    import numpy as np
    from PIL import Image
    import modules.config as cfg
    import modules.flags as flags
    import modules.task_args as task_args
    import modules.xyz_grid as xyz

    op = s['op']
    args, idx = task_args.build(prompt=s['prompt'], negative=s['negative'], seed=s['seed_used'])

    if s['width'] is not None and op == 'gen':
        args[idx['aspect']] = f"{s['width']}×{s['height']}"
    if s['steps'] is not None:
        args[idx['overwrite_step']] = s['steps']

    if s['model']:
        if route == 'remote':
            warnings.append('"model" refused on the remote route: the running instance keeps its '
                            'checkpoint (never swapped under the user\'s feet)')
        else:
            args[idx['base_model']] = _resolve_model(s['model'], list(cfg.model_filenames))

    used_loras = []
    slots = task_args.lora_slot_count(idx)
    if len(s['loras']) > slots:
        warnings.append(f'{len(s["loras"])} LoRAs given, only {slots} slots: extra ones dropped')
    for i, (name, weight) in enumerate(s['loras'][:slots]):
        try:
            resolved = xyz._resolve_lora(name)
        except ValueError as e:
            raise ProtocolError(str(e), 2)
        w = 1.0 if weight is None else weight
        args[idx[f'lora{i}_enabled']] = True
        args[idx[f'lora{i}_name']] = resolved
        args[idx[f'lora{i}_weight']] = w
        used_loras.append(f'{resolved}:{w:g}')

    refs_used = 0
    if s['refs']:
        args[idx['input_image_checkbox']] = True
        args[idx['current_tab']] = 'ip'
        for i, path in enumerate(s['refs']):
            args[idx[f'cn{i}_image']] = np.array(Image.open(path).convert('RGB'))
            args[idx[f'cn{i}_type']] = flags.cn_ip
            refs_used += 1

    engines = list(getattr(flags, 'inpaint_engine_versions', []))
    detail_engine = 'None' if 'None' in engines else args[idx['inpaint_engine']]
    tabs = [name for name in (('face' if s['detail_faces'] else None), ('hand' if s['detail_hands'] else None)) if name]
    n_tabs = sum(1 for k in idx if k.startswith('enh') and k.endswith('_enabled'))
    if len(tabs) > n_tabs:
        warnings.append(f'only {n_tabs} Enhance tab(s) configured: detail_hands dropped')
        tabs = tabs[:n_tabs]
    for i, target in enumerate(tabs):
        # meme reglage que le mode "Improve Detail (face, hand, eyes, etc.)" de l'UI
        args[idx['enhance_checkbox']] = True
        args[idx[f'enh{i}_enabled']] = True
        args[idx[f'enh{i}_mask_prompt']] = target
        args[idx[f'enh{i}_engine']] = detail_engine
        args[idx[f'enh{i}_strength']] = 0.5
        args[idx[f'enh{i}_respective_field']] = 0.0

    if op == 'upscale':
        args[idx['input_image_checkbox']] = True
        args[idx['current_tab']] = 'uov'
        args[idx['uov_input_image']] = np.array(Image.open(s['input']).convert('RGB'))
        denoise = s['denoise']
        if s['factor'] == 1.0:
            args[idx['uov_method']] = flags.subtle_variation
            if denoise is not None:
                args[idx['overwrite_vary_strength']] = denoise
        else:
            if s['factor'] == 2.0 and denoise == 0.0:
                args[idx['uov_method']] = flags.upscale_fast
            else:
                args[idx['uov_method']] = flags.upscale_15 if s['factor'] == 1.5 else flags.upscale_2
                if denoise:
                    args[idx['overwrite_upscale_strength']] = denoise
    elif op == 'inpaint':
        image = Image.open(s['input']).convert('RGB')
        mask = Image.open(s['mask']).convert('L')
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        m = np.array(mask)
        args[idx['input_image_checkbox']] = True
        args[idx['current_tab']] = 'inpaint'
        args[idx['inpaint_input_image']] = {'image': np.array(image), 'mask': np.stack([m, m, m], axis=-1)}
        # prompt LOCAL de la zone : Fooocus le place devant le prompt principal, vide ici
        args[idx['inpaint_additional_prompt']] = s['prompt']
        args[idx['prompt']] = ''
        if s['denoise'] is not None:
            args[idx['inpaint_strength']] = s['denoise']
    return args, {'refs_used': refs_used, 'loras': used_loras}


def execute(args, worker, timeout=3600):
    """Valide le snapshot (un AsyncTask a blanc doit tout consommer), l'enfile dans le
    thread worker de Fooocus et attend la fin."""
    probe = list(args)
    worker.AsyncTask(args=probe)
    if probe:
        raise ProtocolError(f'{len(probe)} ctrls left over: the AsyncTask order changed, update '
                            'modules/task_args.py', 1)
    task = worker.AsyncTask(args=list(args))
    worker.async_tasks.append(task)
    deadline = time.time() + timeout
    while time.time() < deadline:
        while task.yields:
            flag, _ = task.yields.pop(0)
            if flag == 'finish':
                return task
        time.sleep(0.05)
    task.last_stop = 'stop'
    raise ProtocolError(f'no result after {timeout} s', 1)


def run(spec, op, route, worker, timeout=3600):
    """Execute une spec dans CE process (route locale, ou endpoint cache de l'instance)."""
    t0 = time.time()
    s, warnings = validate_spec(spec, op)
    s['seed_used'] = s['seed'] if s['seed'] is not None else random.randint(0, 2 ** 32 - 1)
    args, info = build_task_args(s, route, warnings)
    task = execute(args, worker, timeout)
    images = [r for r in task.results if isinstance(r, str) and os.path.isfile(r)]
    if not images:
        raise ProtocolError('no image file was produced (generation interrupted?)', 1)
    image = images[-1]  # avec Enhance, la derniere est la version detaillee
    if s['out_dir']:
        os.makedirs(s['out_dir'], exist_ok=True)
        dst = os.path.join(s['out_dir'], os.path.basename(image))
        shutil.copy2(image, dst)
        image = dst
    return {'ok': True, 'protocol': PROTOCOL, 'tool': TOOL, 'version': tool_version(),
            'route': route, 'images': [os.path.abspath(image)], 'seed_used': s['seed_used'],
            'refs_used': info['refs_used'], 'loras': info['loras'],
            'timings': {'total_s': round(time.time() - t0, 2)}, 'warnings': warnings}


# ------------------------------------------------------------------- caps

def caps_dict(models, loras, model_loaded, instance):
    return {'ok': True, 'protocol': PROTOCOL, 'tool': TOOL, 'version': tool_version(),
            'ops': list(OPS), 'models': list(models), 'loras': list(loras),
            'model_loaded': bool(model_loaded),
            'supports': {'loras': True, 'refs': True, 'max_refs': MAX_REFS, 'seed': True,
                         'negative': True, 'arbitrary_size': True, 'faces': False,
                         'detail_faces': True, 'detail_hands': True, 'edit': False,
                         'inpaint': True, 'img2img': True},
            'instance': instance}


def _model_loaded():
    pipeline = sys.modules.get('modules.default_pipeline')
    return bool(pipeline is not None and getattr(pipeline, 'model_base', None) is not None
                and getattr(pipeline.model_base, 'unet_with_lora', None) is not None)


# ------------------------------------------- endpoints caches (webui.py)

def remote_caps(_unused=None):
    """Endpoint cache cli_caps : capacites de CETTE instance, en JSON."""
    import modules.config as cfg
    return json.dumps(caps_dict(cfg.model_filenames, cfg.lora_filenames, _model_loaded(),
                                {'running': True, 'url': instance_url(), 'tool': TOOL,
                                 'version': tool_version()}))


def remote_run(spec_json):
    """Endpoint cache cli_gen : execute la spec (son champ op) dans CETTE instance."""
    try:
        spec = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
        op = str((spec or {}).get('op') or 'gen')
        import modules.async_worker as worker
        res = run(spec, op, 'remote', worker)
    except ProtocolError as e:
        res = {'ok': False, 'error': str(e), 'code': e.code}
    except Exception as e:
        res = {'ok': False, 'error': f'{type(e).__name__}: {e}', 'code': 1}
    return json.dumps(res)


# ------------------------------------------------------------------ routes

def instance_url():
    env = os.environ.get('FOOOCUS_CLI_URL')
    if env:
        return env.rstrip('/')
    try:
        with open(os.path.join(ROOT, 'config.txt'), encoding='utf-8') as f:
            url = str(((json.load(f) or {}).get('cli_protocol') or {}).get('instance_url') or '').strip()
        if url:
            return url.rstrip('/')
    except Exception:
        pass
    return 'http://127.0.0.1:%s' % (os.environ.get('GRADIO_SERVER_PORT') or '7865')


def gradio_call(url, name, payload, timeout):
    """POST <url>/run/<name> (Gradio 3.41, evenement hors file). Renvoie la 1re sortie,
    decodee si c'est du JSON."""
    body = json.dumps({'data': [payload]}).encode('utf-8')
    req = urllib.request.Request(f'{url.rstrip("/")}/run/{name}', data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode('utf-8'))
    data = (resp or {}).get('data') or [None]
    out = data[0]
    if isinstance(out, str):
        try:
            return json.loads(out)
        except ValueError:
            return out
    return out


def probe_instance(url, timeout=4):
    """Les caps de l'instance si c'est BIEN Fooocus2026 qui repond, sinon None."""
    try:
        caps = gradio_call(url, 'cli_caps', '{}', timeout)
    except Exception:
        return None
    return caps if isinstance(caps, dict) and caps.get('tool') == TOOL else None


def _load_config():
    """Config + listes de modeles, SANS le thread worker : caps ne charge ni GPU ni modele
    (protocole §5). update_files() scanne les dossiers (architectures en cache)."""
    sys.argv = [sys.argv[0]]  # args_manager analyse sys.argv a l'import
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import modules.config as cfg
    cfg.update_files()
    return cfg


def _load_backend():
    """Route locale : charge Fooocus dans ce process, comme xyz_cli.py."""
    cfg = _load_config()
    import modules.async_worker as worker
    return worker, cfg


def _read_spec(path):
    if path == '-':
        text = sys.stdin.buffer.read().decode('utf-8-sig')
    else:
        with open(caller_path(path), encoding='utf-8-sig') as f:
            text = f.read()
    try:
        return json.loads(text)
    except ValueError as e:
        raise ProtocolError(f'the spec is not valid JSON: {e}', 2)


def main(argv=None, out=None):
    out = out or sys.stdout
    argv = list(sys.argv[1:] if argv is None else argv)

    def emit(payload, code):
        out.write(json.dumps(payload, ensure_ascii=False) + '\n')
        out.flush()
        return code

    cmd = argv[0] if argv else ''
    opts = {'--spec': None, '--remote': None, '--timeout': '3600'}
    local = '--local' in argv
    rest = [a for a in argv[1:] if a != '--local']
    i = 0
    while i < len(rest):
        if rest[i] in opts and i + 1 < len(rest):
            opts[rest[i]] = rest[i + 1]
            i += 2
        else:
            return emit({'ok': False, 'error': f'unexpected argument "{rest[i]}"'}, 2)
    if cmd not in OPS and cmd != 'edit':
        return emit({'ok': False, 'error': f'unknown command "{cmd}" (caps, gen, upscale, inpaint)'}, 3)
    try:
        timeout = int(opts['--timeout'])
    except ValueError:
        return emit({'ok': False, 'error': '--timeout must be an integer (seconds)'}, 2)
    url = (opts['--remote'] or instance_url()).rstrip('/')

    # stdout reserve a la ligne JSON : tout ce que Fooocus imprime part sur stderr
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        if cmd == 'caps':
            inst = None if local else probe_instance(url)
            if inst:
                inst['route'] = 'remote'
                return emit(inst, 0)
            if opts['--remote']:
                return emit({'ok': False, 'error': f'no Fooocus2026 instance answers at {url}'}, 4)
            cfg = _load_config()
            caps = caps_dict(cfg.model_filenames, cfg.lora_filenames, False,
                             {'running': False, 'url': url})
            caps['route'] = 'local'
            return emit(caps, 0)

        if not opts['--spec']:
            return emit({'ok': False, 'error': f'{cmd} needs --spec <file> (or --spec - for stdin)'}, 2)
        spec = _read_spec(opts['--spec'])
        validate_spec(spec, cmd)  # erreurs de spec avant toute route ou chargement

        if not local and probe_instance(url):
            remote_spec = dict(spec, op=cmd)
            for key in ('input', 'mask', 'out_dir'):
                if remote_spec.get(key):
                    remote_spec[key] = caller_path(str(remote_spec[key]))
            if remote_spec.get('refs'):
                remote_spec['refs'] = [caller_path(str(r)) for r in remote_spec['refs']]
            try:
                res = gradio_call(url, 'cli_gen', json.dumps(remote_spec), timeout)
            except Exception as e:
                return emit({'ok': False, 'error': f'remote call to {url} failed: {e}'}, 1)
            if not isinstance(res, dict):
                return emit({'ok': False, 'error': f'unexpected reply from {url}: {str(res)[:200]}'}, 1)
            return emit(res, 0 if res.get('ok') else int(res.pop('code', 1)))
        if opts['--remote']:
            return emit({'ok': False, 'error': f'no Fooocus2026 instance answers at {url}'}, 4)

        worker, _cfg = _load_backend()
        return emit(run(spec, cmd, 'local', worker, timeout), 0)
    except ProtocolError as e:
        return emit({'ok': False, 'error': str(e)}, e.code)
    except Exception as e:
        return emit({'ok': False, 'error': f'{type(e).__name__}: {e}'}, 1)
    finally:
        sys.stdout = real_stdout


if __name__ == '__main__':
    code = main()
    sys.stdout.flush()
    os._exit(code)  # le thread worker de Fooocus est un daemon : sortie immediate
