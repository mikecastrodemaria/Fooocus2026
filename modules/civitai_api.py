"""
CivitAI API Integration for Fooocus
Fetches community-recommended settings for models from CivitAI.

Strategy:
  1. Hash the local model file (SHA256, via existing hash_cache)
  2. Look up the model on CivitAI by hash -> get modelVersionId
  3. Fetch top-rated images for that model version
  4. Analyze the generation metadata (meta) to extract consensus settings
"""

import json
import os
import threading
from collections import Counter
from statistics import median
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from urllib.parse import urlencode, quote

from modules.hash_cache import sha256_from_cache
from modules.util import get_file_from_folder_list, calculate_sha256

CIVITAI_API_BASE = 'https://civitai.com/api/v1'
REQUEST_TIMEOUT = 15  # seconds

# Cache for full SHA256 hashes (CivitAI needs the full 64-char hash, not Fooocus's truncated 10-char)
_full_hash_cache = {}


def _cache_dir():
    """Resolve the CivitAI cache dir lazily so a config.txt change is honoured
    without re-importing this module. Defaults to the historical
    ./civitai_cache for full back-compat with installs that don't set
    path_civitai_cache.
    """
    try:
        from modules import config
        path = getattr(config, 'path_civitai_cache', None)
        if path:
            return os.path.abspath(path)
    except Exception:
        pass
    return os.path.abspath('./civitai_cache')


# Legacy module-level constant kept for any external code that imports it.
# Computed once at first import; in normal usage modules.config is already
# loaded by the time civitai_api is first imported (config -> civitai_api).
CIVITAI_CACHE_DIR = _cache_dir()


def _get_cache_path(model_filename):
    """Get the local cache file path for a model's CivitAI settings."""
    safe_name = os.path.splitext(os.path.basename(model_filename))[0]
    return os.path.join(_cache_dir(), f'{safe_name}.civitai.json')


def load_cached_settings(model_filename):
    """Load cached CivitAI settings for a model from local disk.

    Returns:
        Full result dict (with model_info + settings) or None if no cache
    """
    cache_path = _get_cache_path(model_filename)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f'[CivitAI] Loaded cached settings for {model_filename}')
            data['_from_cache'] = True
            return data
        except Exception as e:
            print(f'[CivitAI] Cache read error for {model_filename}: {e}')
    return None


def save_settings_to_cache(model_filename, result):
    """Save CivitAI settings to local cache.

    Args:
        model_filename: Name of the model file
        result: Full result dict from fetch_recommended_settings()
    """
    if 'error' in result and 'settings' not in result:
        return  # Don't cache errors

    try:
        os.makedirs(_cache_dir(), exist_ok=True)
        cache_path = _get_cache_path(model_filename)
        # Remove internal keys before saving
        to_save = {k: v for k, v in result.items() if not k.startswith('_')}
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(to_save, f, indent=2, ensure_ascii=False)
        print(f'[CivitAI] Cached settings for {model_filename}')
    except Exception as e:
        print(f'[CivitAI] Cache write error: {e}')


def _get_full_sha256(filepath):
    """Get the full SHA256 hash (64 chars) for CivitAI lookup.

    Fooocus's built-in hash cache truncates to 10 chars which is not enough for CivitAI.
    We maintain our own cache for the full hash.
    """
    if filepath in _full_hash_cache:
        return _full_hash_cache[filepath]

    print(f'[CivitAI] Calculating full SHA256 for {os.path.basename(filepath)}...')
    full_hash = calculate_sha256(filepath)
    _full_hash_cache[filepath] = full_hash
    print(f'[CivitAI] Full SHA256: {full_hash}')
    return full_hash

# Sampler name mapping: CivitAI uses A1111-style names, Fooocus uses ldm_patched names
SAMPLER_MAP_TO_FOOOCUS = {
    'DPM++ 2M Karras': ('dpmpp_2m_sde_gpu', 'karras'),
    'DPM++ 2M SDE Karras': ('dpmpp_2m_sde_gpu', 'karras'),
    'DPM++ 2M SDE': ('dpmpp_2m_sde_gpu', 'normal'),
    'DPM++ SDE Karras': ('dpmpp_sde_gpu', 'karras'),
    'DPM++ SDE': ('dpmpp_sde_gpu', 'normal'),
    'DPM++ 2S a Karras': ('dpmpp_2s_ancestral', 'karras'),
    'DPM++ 2S a': ('dpmpp_2s_ancestral', 'normal'),
    'DPM++ 3M SDE Karras': ('dpmpp_3m_sde_gpu', 'karras'),
    'DPM++ 3M SDE': ('dpmpp_3m_sde_gpu', 'normal'),
    'DPM++ 3M SDE Exponential': ('dpmpp_3m_sde_gpu', 'exponential'),
    'Euler': ('euler', 'normal'),
    'Euler a': ('euler_ancestral', 'normal'),
    'Heun': ('heun', 'normal'),
    'LMS': ('lms', 'normal'),
    'LMS Karras': ('lms', 'karras'),
    'DDIM': ('ddim', 'ddim_uniform'),
    'UniPC': ('uni_pc', 'normal'),
}


def _api_request(endpoint, params=None, api_key=None):
    """Make a GET request to the CivitAI API.

    Args:
        endpoint: API path (e.g., '/models/12345')
        params: Optional dict of query parameters
        api_key: Optional CivitAI API key

    Returns:
        Parsed JSON response or None on error
    """
    if params is None:
        params = {}

    # CivitAI accepts the token as query param (more reliable than Bearer for some endpoints)
    if api_key:
        params['token'] = api_key

    url = f'{CIVITAI_API_BASE}{endpoint}'
    if params:
        url += '?' + urlencode(params, quote_via=quote)

    headers = {
        'Content-Type': 'application/json',
        'User-Agent': 'Fooocus/2.5.5 (CivitAI-Integration)',
    }

    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            return json.loads(response.read().decode('utf-8'))
    except HTTPError as e:
        body = ''
        try:
            body = e.read().decode('utf-8', errors='ignore')[:200]
        except Exception:
            pass
        print(f'[CivitAI] HTTP Error {e.code}: {e.reason} for {endpoint} | {body}')
        return None
    except URLError as e:
        print(f'[CivitAI] URL Error: {e.reason} for {endpoint}')
        return None
    except Exception as e:
        print(f'[CivitAI] Request failed: {e}')
        return None


def get_model_version_by_hash(file_hash, api_key=None):
    """Look up a model version on CivitAI by SHA256 hash.

    Returns:
        dict with modelId, modelVersionId, modelName, versionName, baseModel,
        trainedWords (list), or None on miss.
    """
    # CivitAI expects uppercase hash, first 10 chars is enough but full is better
    data = _api_request(f'/model-versions/by-hash/{file_hash}', api_key=api_key)
    if data and 'id' in data:
        triggers = [str(w).strip() for w in (data.get('trainedWords') or []) if str(w).strip()]
        return {
            'modelId': data.get('modelId'),
            'modelVersionId': data.get('id'),
            'modelName': data.get('model', {}).get('name', 'Unknown'),
            'versionName': data.get('name', 'Unknown'),
            'baseModel': data.get('baseModel', 'Unknown'),
            'trainedWords': triggers,
        }
    return None


def get_top_images(model_version_id, api_key=None, limit=20):
    """Fetch top-rated images for a model version.

    Returns:
        List of image metadata dicts
    """
    params = {
        'modelVersionId': model_version_id,
        'sort': 'Most Reactions',
        'limit': limit,
    }
    data = _api_request('/images', params=params, api_key=api_key)
    if data and 'items' in data:
        return data['items']
    return []


def analyze_image_settings(images):
    """Analyze generation metadata from multiple images to find consensus settings.

    Args:
        images: List of image dicts from CivitAI API

    Returns:
        dict with recommended settings and confidence stats
    """
    samplers = []
    cfg_scales = []
    steps_list = []
    clip_skips = []
    sizes = []
    total_with_meta = 0

    for img in images:
        meta = img.get('meta')
        if not meta or not isinstance(meta, dict):
            continue
        total_with_meta += 1

        # Sampler
        sampler = meta.get('sampler')
        if sampler:
            samplers.append(sampler)

        # CFG Scale
        cfg = meta.get('cfgScale')
        if cfg is not None:
            try:
                cfg_scales.append(float(cfg))
            except (ValueError, TypeError):
                pass

        # Steps
        s = meta.get('steps')
        if s is not None:
            try:
                steps_list.append(int(s))
            except (ValueError, TypeError):
                pass

        # Clip Skip
        clip = meta.get('Clip skip') or meta.get('clip_skip') or meta.get('clipSkip')
        if clip is not None:
            try:
                clip_skips.append(int(clip))
            except (ValueError, TypeError):
                pass

        # Size
        size = meta.get('Size')
        if size:
            sizes.append(size)

    if total_with_meta == 0:
        return None

    result = {
        'total_images_analyzed': total_with_meta,
    }

    # Sampler consensus
    if samplers:
        counter = Counter(samplers)
        top_sampler, top_count = counter.most_common(1)[0]
        result['sampler_civitai'] = top_sampler
        result['sampler_confidence'] = round(top_count / len(samplers) * 100)

        # Map to Fooocus names
        if top_sampler in SAMPLER_MAP_TO_FOOOCUS:
            fooocus_sampler, fooocus_scheduler = SAMPLER_MAP_TO_FOOOCUS[top_sampler]
            result['sampler_fooocus'] = fooocus_sampler
            result['scheduler_fooocus'] = fooocus_scheduler
        else:
            result['sampler_fooocus'] = None
            result['scheduler_fooocus'] = None

    # CFG consensus
    if cfg_scales:
        result['cfg_scale'] = round(median(cfg_scales), 1)
        result['cfg_range'] = (round(min(cfg_scales), 1), round(max(cfg_scales), 1))

    # Steps consensus
    if steps_list:
        result['steps'] = int(median(steps_list))
        result['steps_range'] = (min(steps_list), max(steps_list))

    # Clip Skip consensus
    if clip_skips:
        counter = Counter(clip_skips)
        top_clip, top_count = counter.most_common(1)[0]
        result['clip_skip'] = top_clip
        result['clip_skip_confidence'] = round(top_count / len(clip_skips) * 100)

    # Resolution consensus
    if sizes:
        counter = Counter(sizes)
        top_size, _ = counter.most_common(1)[0]
        result['resolution'] = top_size

    return result


def fetch_recommended_settings(model_filename, paths_checkpoints, api_key=None, progress_callback=None, force_refresh=False):
    """Full pipeline: hash model -> find on CivitAI -> analyze community settings.

    Args:
        model_filename: Name of the model file (e.g., 'juggernautXL_v8Rundiffusion.safetensors')
        paths_checkpoints: List of checkpoint directories to search
        api_key: Optional CivitAI API key
        progress_callback: Optional callable(step, message) for progress updates
        force_refresh: If True, skip cache and fetch fresh from CivitAI

    Returns:
        dict with model_info and settings, or error dict
    """
    def _progress(step, msg):
        if progress_callback:
            progress_callback(step, msg)
        print(f'[CivitAI] {msg}')

    # Step 0: Check local cache first
    if not force_refresh:
        cached = load_cached_settings(model_filename)
        if cached and 'settings' in cached:
            _progress(0, f'Loaded settings from local cache for {model_filename}')
            return cached

    # Step 1: Find the file
    _progress(1, f'Locating {model_filename}...')
    try:
        filepath = get_file_from_folder_list(model_filename, paths_checkpoints)
        if not os.path.isfile(filepath):
            return {'error': f'Model file not found: {model_filename}'}
    except Exception as e:
        return {'error': f'Error locating model: {str(e)}'}

    # Step 2: Get full SHA256 hash (CivitAI needs all 64 chars, not Fooocus's truncated 10)
    _progress(2, 'Calculating full SHA256 hash (may take a moment on first run)...')
    try:
        file_hash = _get_full_sha256(filepath)
    except Exception as e:
        return {'error': f'Error hashing model: {str(e)}'}

    if not file_hash:
        return {'error': 'Could not calculate file hash.'}

    # Step 3: Look up on CivitAI
    _progress(3, f'Looking up hash {file_hash[:10]}... on CivitAI...')
    model_info = get_model_version_by_hash(file_hash, api_key=api_key)

    if not model_info:
        return {'error': f'Model not found on CivitAI (hash: {file_hash[:10]}...). '
                         f'It may not be uploaded there, or the hash format differs.'}

    # Step 4: Fetch top images
    _progress(4, f'Fetching top images for {model_info["modelName"]} ({model_info["versionName"]})...')
    images = get_top_images(model_info['modelVersionId'], api_key=api_key, limit=20)

    if not images:
        return {
            'model_info': model_info,
            'error': 'No images found for this model version on CivitAI.'
        }

    # Step 5: Analyze
    _progress(5, f'Analyzing settings from {len(images)} images...')
    settings = analyze_image_settings(images)

    if not settings:
        return {
            'model_info': model_info,
            'error': 'No generation metadata found in the images.'
        }

    result = {
        'model_info': model_info,
        'settings': settings,
    }

    # Save to local cache for offline / instant reload
    save_settings_to_cache(model_filename, result)

    return result


def format_settings_html(result):
    """Format the analysis result as an HTML panel for the Gradio UI.

    Args:
        result: dict from fetch_recommended_settings()

    Returns:
        HTML string
    """
    if 'error' in result and 'model_info' not in result:
        return f'<div style="padding:10px;border:1px solid #ff6b6b;border-radius:8px;background:#2d1b1b;">' \
               f'<b style="color:#ff6b6b;">CivitAI Lookup Failed</b><br/>{result["error"]}</div>'

    model_info = result.get('model_info', {})
    model_name = model_info.get('modelName', '?')
    version_name = model_info.get('versionName', '?')
    base_model = model_info.get('baseModel', '?')

    if 'error' in result:
        return f'<div style="padding:10px;border:1px solid #ffa500;border-radius:8px;background:#2d2510;">' \
               f'<b style="color:#ffa500;">Found: {model_name} ({version_name})</b> [{base_model}]<br/>' \
               f'{result["error"]}</div>'

    settings = result['settings']
    total = settings.get('total_images_analyzed', 0)
    from_cache = result.get('_from_cache', False)

    cache_badge = ''
    if from_cache:
        cache_badge = (' <span style="background:#2a4a3a;color:#6fcf97;padding:2px 8px;'
                       'border-radius:4px;font-size:11px;margin-left:8px;">cached</span>')

    trained_words = model_info.get('trainedWords') or []
    triggers_block = ''
    if trained_words:
        joined = ', '.join(trained_words)
        triggers_block = (
            f'<div style="margin:6px 0 10px 0;padding:6px 8px;background:#13302c;'
            f'border-left:3px solid #4ecdc4;border-radius:4px;font-size:13px;">'
            f'<b style="color:#4ecdc4;">Triggers:</b> '
            f'<span style="color:#ddd;">{joined}</span>'
            f'</div>'
        )

    rows = []

    # Sampler
    if 'sampler_civitai' in settings:
        sampler_display = settings['sampler_civitai']
        fooocus_sampler = settings.get('sampler_fooocus', '?')
        fooocus_sched = settings.get('scheduler_fooocus', '?')
        conf = settings.get('sampler_confidence', 0)
        rows.append(f'<tr><td><b>Sampler</b></td><td>{sampler_display}</td>'
                     f'<td style="color:#888;">{fooocus_sampler} + {fooocus_sched}</td>'
                     f'<td>{conf}%</td></tr>')

    # CFG
    if 'cfg_scale' in settings:
        cfg_range = settings.get('cfg_range', ('?', '?'))
        rows.append(f'<tr><td><b>CFG Scale</b></td><td>{settings["cfg_scale"]}</td>'
                     f'<td style="color:#888;">range: {cfg_range[0]}-{cfg_range[1]}</td>'
                     f'<td>median</td></tr>')

    # Steps
    if 'steps' in settings:
        steps_range = settings.get('steps_range', ('?', '?'))
        rows.append(f'<tr><td><b>Steps</b></td><td>{settings["steps"]}</td>'
                     f'<td style="color:#888;">range: {steps_range[0]}-{steps_range[1]}</td>'
                     f'<td>median</td></tr>')

    # Clip Skip
    if 'clip_skip' in settings:
        conf = settings.get('clip_skip_confidence', 0)
        rows.append(f'<tr><td><b>Clip Skip</b></td><td>{settings["clip_skip"]}</td>'
                     f'<td></td><td>{conf}%</td></tr>')

    # Resolution
    if 'resolution' in settings:
        rows.append(f'<tr><td><b>Resolution</b></td><td>{settings["resolution"]}</td>'
                     f'<td></td><td>top</td></tr>')

    table_rows = '\n'.join(rows)

    html = f'''<div style="padding:12px;border:1px solid #4ecdc4;border-radius:8px;background:#1a2d2b;">
  <div style="margin-bottom:8px;">
    <b style="color:#4ecdc4;font-size:14px;">CivitAI Community Settings</b>{cache_badge}<br/>
    <span style="color:#aaa;">{model_name} ({version_name}) [{base_model}] &mdash; {total} images analyzed</span>
  </div>
  {triggers_block}
  <table style="width:100%;border-collapse:collapse;font-size:13px;">
    <tr style="border-bottom:1px solid #333;">
      <th style="text-align:left;padding:4px;color:#888;">Setting</th>
      <th style="text-align:left;padding:4px;color:#888;">Value</th>
      <th style="text-align:left;padding:4px;color:#888;">Details</th>
      <th style="text-align:left;padding:4px;color:#888;">Confidence</th>
    </tr>
    {table_rows}
  </table>
</div>'''

    return html


# =============================================================================
# LoRA trigger words (custom-3)
# =============================================================================

_CACHE_SUFFIX_BY_KIND = {
    'lora': 'lora',
    'embedding': 'embedding',
}


def _get_triggers_cache_path(filename, kind='lora'):
    """Local cache file for a model's trigger words, namespaced by kind."""
    safe_name = os.path.splitext(os.path.basename(filename))[0]
    suffix = _CACHE_SUFFIX_BY_KIND.get(kind, kind)
    return os.path.join(_cache_dir(), f'{safe_name}.{suffix}.civitai.json')


def _get_lora_cache_path(lora_filename):
    """Legacy name kept for any external callers."""
    return _get_triggers_cache_path(lora_filename, kind='lora')


def load_cached_triggers(filename, kind='lora'):
    path = _get_triggers_cache_path(filename, kind=kind)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f'[CivitAI] {kind} cache read error for {filename}: {e}')
    return None


def save_triggers_to_cache(filename, data, kind='lora'):
    try:
        os.makedirs(_cache_dir(), exist_ok=True)
        path = _get_triggers_cache_path(filename, kind=kind)
        to_save = {k: v for k, v in data.items() if not k.startswith('_')}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(to_save, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f'[CivitAI] {kind} cache write error: {e}')


# Legacy names for backwards-compat
def load_cached_lora_triggers(lora_filename):
    return load_cached_triggers(lora_filename, kind='lora')


def save_lora_triggers_to_cache(lora_filename, data):
    save_triggers_to_cache(lora_filename, data, kind='lora')


def fetch_model_triggers(filename, paths, kind='lora', api_key=None, force_refresh=False):
    """Fetch trainedWords for a model (LoRA or embedding) from CivitAI by hash, cached.

    Args:
        filename: file name (e.g., 'detail_tweaker_xl.safetensors').
        paths: list of search directories (e.g., modules.config.paths_loras
               or [modules.config.path_embeddings]).
        kind: 'lora' or 'embedding' — only affects the cache namespace.
        api_key: optional CivitAI API key.
        force_refresh: if True, skip cache and re-query CivitAI.

    Returns:
        {'model_info': ..., 'trainedWords': [...]} on success,
        {'error': ...} on any failure. Misses are cached too.
    """
    if not filename or filename == 'None':
        return {'error': f'No {kind} selected.'}

    if not force_refresh:
        cached = load_cached_triggers(filename, kind=kind)
        if cached is not None:
            cached['_from_cache'] = True
            return cached

    try:
        filepath = get_file_from_folder_list(filename, paths)
        if not os.path.isfile(filepath):
            return {'error': f'{kind} file not found: {filename}'}
    except Exception as e:
        return {'error': f'Error locating {kind}: {e}'}

    try:
        file_hash = _get_full_sha256(filepath)
    except Exception as e:
        return {'error': f'Error hashing {kind}: {e}'}

    if not file_hash:
        return {'error': f'Could not calculate {kind} hash.'}

    data = _api_request(f'/model-versions/by-hash/{file_hash}', api_key=api_key)
    if not data or 'id' not in data:
        miss = {'error': f'{kind} not on CivitAI (hash {file_hash[:10]}...).'}
        save_triggers_to_cache(filename, miss, kind=kind)
        return miss

    info = {
        'modelId': data.get('modelId'),
        'modelVersionId': data.get('id'),
        'modelName': data.get('model', {}).get('name', 'Unknown'),
        'versionName': data.get('name', 'Unknown'),
        'baseModel': data.get('baseModel', 'Unknown'),
    }
    triggers = [str(w).strip() for w in (data.get('trainedWords') or []) if str(w).strip()]

    result = {'model_info': info, 'trainedWords': triggers}
    save_triggers_to_cache(filename, result, kind=kind)
    return result


def fetch_lora_triggers(lora_filename, paths_loras, api_key=None, force_refresh=False):
    """Legacy shim — use fetch_model_triggers(kind='lora') instead."""
    return fetch_model_triggers(
        filename=lora_filename, paths=paths_loras, kind='lora',
        api_key=api_key, force_refresh=force_refresh,
    )


def format_lora_triggers_display(result):
    """Turn a fetch_lora_triggers() / fetch_lora_triggers_combined() result into
    a user-facing trigger string for a read-only Textbox.

    If the result contains no usable triggers, returns a parenthesised
    placeholder so the "Copy to prompt" handler can detect and skip it.
    """
    if not result:
        return '(no data)'
    # Combined result: merged triggers list is authoritative
    if 'merged' in result:
        merged = result.get('merged') or []
        if merged:
            return ', '.join(merged)
        # No merged triggers — surface why
        local_err = result.get('local', {}).get('error', '')
        civ_err = result.get('civitai', {}).get('error', '')
        bits = []
        if local_err:
            bits.append(f'local: {local_err}')
        if civ_err:
            bits.append(f'civitai: {civ_err}')
        return '(no triggers found — ' + '; '.join(bits) + ')' if bits else '(no triggers found)'
    # Legacy single-source result shape
    if 'error' in result:
        return f'(no triggers — {result["error"]})'
    info = result.get('model_info') or {}
    words = result.get('trainedWords') or []
    if not words:
        name = info.get('modelName', '?')
        return f'({name} — no trigger words listed)'
    return ', '.join(words)


def fetch_model_triggers_combined(filename, paths, kind='lora', api_key=None, force_refresh=False):
    """Merge triggers from local safetensors metadata and CivitAI.

    Works for both LoRAs (kind='lora') and textual inversion embeddings
    (kind='embedding'). Local triggers appear first (ground truth from training),
    CivitAI-only extras are appended and deduped.

    Args:
        filename: file basename (e.g., 'detail_tweaker_xl.safetensors').
        paths: list of search directories for that kind.
        kind: 'lora' or 'embedding'.

    Returns:
        {
          'local':    <result from lora_metadata.get_*_triggers_from_file>,
          'civitai':  <result from fetch_model_triggers>,
          'merged':   [...],                  # deduped, ordered: local first
          'model_info': <civitai model_info or None>,
          'sources':  ['local', 'civitai'],   # which actually contributed
          'kind':     'lora' | 'embedding',
        }
    """
    from modules import lora_metadata

    if kind == 'embedding':
        local = lora_metadata.get_embedding_triggers_from_file(filename, paths)
    else:
        local = lora_metadata.get_lora_triggers_from_file(filename, paths)

    civitai = fetch_model_triggers(
        filename=filename, paths=paths, kind=kind,
        api_key=api_key, force_refresh=force_refresh,
    )

    merged = []
    seen = set()
    sources = []

    for w in (local.get('trainedWords') or []):
        wl = w.lower()
        if wl not in seen:
            merged.append(w)
            seen.add(wl)
    if merged:
        sources.append('local')

    civ_added = 0
    for w in (civitai.get('trainedWords') or []):
        wl = w.lower()
        if wl not in seen:
            merged.append(w)
            seen.add(wl)
            civ_added += 1
    if civ_added:
        sources.append('civitai')

    return {
        'local': local,
        'civitai': civitai,
        'merged': merged,
        'model_info': civitai.get('model_info'),
        'sources': sources,
        'kind': kind,
    }


def fetch_lora_triggers_combined(lora_filename, paths_loras, api_key=None, force_refresh=False):
    """Legacy shim — use fetch_model_triggers_combined(kind='lora') instead."""
    return fetch_model_triggers_combined(
        filename=lora_filename, paths=paths_loras, kind='lora',
        api_key=api_key, force_refresh=force_refresh,
    )
