"""custom-8 (Asset Browser) — model indexer.

Walks `models/loras/`, `models/checkpoints/` (multiple paths), and
`models/embeddings/` to build the JSON manifests consumed by the SPA tabs:

- `outputs/_index/loras.json`
- `outputs/_index/checkpoints.json`
- `outputs/_index/embeddings.json`

Re-uses existing helpers:
- `modules.lora_metadata.get_lora_triggers_from_file` / `get_embedding_triggers_from_file`
- `modules.civitai_api.load_cached_triggers` (cache-only, NO API calls)
- `modules.civitai_api.load_cached_settings` (cache-only, NO API calls)

Designed for hundreds of models on a fast SSD: every path is touched at
most once, no network, thumbnails are written under `outputs/_previews/` and
re-used across reindexes (invalidation by source mtime).

**M2b ships the real implementation.** Toggle still respected.
"""
import datetime
import hashlib
import json
import os
import threading

import modules.config
from modules.util import get_file_from_folder_list

# Sidecar previews from CivitAI sometimes ship huge — disable Pillow's
# decompression-bomb warning/error so the indexer doesn't spam the console
# nor refuse to thumbnail oversize previews. We trust our local model dir.
try:
    from PIL import Image as _PILImage
    _PILImage.MAX_IMAGE_PIXELS = None
except Exception:
    pass


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

INDEX_DIR_NAME = '_index'
PREVIEWS_DIR_NAME = '_previews'
PLACEHOLDER_DIR_NAME = 'placeholders'

# Placeholder PNGs are generated at this fixed size (independent of
# thumbnail_size) so the PhotoSwipe lightbox shows them at a usable size
# instead of stretched-up 256x256 squares. The 256x256 thumbnail used in
# the grid is derived from this image via _make_preview_thumb.
PLACEHOLDER_FULL_SIZE = 1024


def _thumb_size() -> int:
    return int(modules.config.asset_browser_setting('thumbnail_size', 256))


def _thumb_quality() -> int:
    return int(modules.config.asset_browser_setting('thumbnail_quality', 85))


def _placeholder_label_max() -> int:
    return int(modules.config.asset_browser_setting('placeholder_label_max', 24))

# Lookup order for sidecar previews (A1111 / ComfyUI compatible).
PREVIEW_SUFFIXES = [
    '.preview.png', '.preview.jpg', '.preview.jpeg',
    '.png', '.jpg', '.jpeg',
    '_preview.png',
]

# Heuristic for negative-style embeddings.
NEGATIVE_PREFIXES = ('neg', 'bad', 'unaesthetic', 'fast_neg', 'fast-neg', 'easyneg')

_lock = threading.Lock()


# --------------------------------------------------------------------------
# Toggle helpers
# --------------------------------------------------------------------------

def _enabled() -> bool:
    return modules.config.asset_browser_enabled()


def _index_on_boot_enabled() -> bool:
    return _enabled() and bool(
        modules.config.asset_browser_setting('index_models_on_boot', True)
    )


def _outputs_root() -> str:
    return modules.config.path_outputs


def _index_dir() -> str:
    return os.path.join(_outputs_root(), INDEX_DIR_NAME)


def _previews_dir(kind: str) -> str:
    """outputs/_previews/<kind>/  — thumbnails of sidecar previews + placeholders."""
    return os.path.join(_outputs_root(), PREVIEWS_DIR_NAME, kind)


# --------------------------------------------------------------------------
# Preview discovery + placeholder generation
# --------------------------------------------------------------------------

def _find_sidecar_preview(model_filepath: str) -> str:
    """Return the first matching sidecar preview path next to the model file, or ''.

    Tries the 5 conventional suffixes in order.
    """
    if not model_filepath:
        return ''
    base, _ = os.path.splitext(model_filepath)
    for suffix in PREVIEW_SUFFIXES:
        candidate = base + suffix
        if os.path.isfile(candidate):
            return candidate
    return ''


def _hash_id(s: str, length: int = 12) -> str:
    """Stable short hash for cache keys."""
    return hashlib.sha1(s.encode('utf-8', errors='replace')).hexdigest()[:length]


def _make_placeholder_png(filename_for_label: str, dest_path: str) -> bool:
    """Generate a hash-derived gradient placeholder PNG with the filename overlay.
    Always rendered at PLACEHOLDER_FULL_SIZE (1024) regardless of thumbnail_size,
    so the lightbox shows a usable full-screen image. The 256x256 grid thumb is
    derived from this PNG via _make_preview_thumb. Idempotent.
    """
    try:
        if os.path.isfile(dest_path):
            return True
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        h = hashlib.sha1(filename_for_label.encode('utf-8', errors='replace')).hexdigest()
        c1 = np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float32)
        c2 = np.array([int(h[6:8], 16), int(h[8:10], 16), int(h[10:12], 16)], dtype=np.float32)

        size = PLACEHOLDER_FULL_SIZE
        # Vectorised vertical gradient — putpixel on 1024x1024 = ~1M iterations
        # in Python = several seconds per placeholder. numpy does it in <5 ms.
        t = np.linspace(0, 1, size, dtype=np.float32).reshape(size, 1, 1)   # (H, 1, 1)
        gradient = (c1 * (1 - t) + c2 * t).astype(np.uint8)                  # (H, 1, 3)
        arr = np.broadcast_to(gradient, (size, size, 3)).copy()              # (H, W, 3)
        im = Image.fromarray(arr, mode='RGB')

        draw = ImageDraw.Draw(im)
        label = os.path.splitext(os.path.basename(filename_for_label))[0]
        max_len = _placeholder_label_max()
        if len(label) > max_len:
            head = max(4, (max_len - 1) // 2)
            tail = max(4, max_len - head - 1)
            label = label[:head] + '…' + label[-tail:]
        font_px = max(16, size // 18)   # 56 px on a 1024 placeholder, scales linearly
        try:
            font = ImageFont.truetype('arial.ttf', font_px)
        except Exception:
            font = ImageFont.load_default()
        try:
            bbox = draw.textbbox((0, 0), label, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = (size // 2, font_px)
        tx = max(8, (size - tw) // 2)
        pad_y = max(8, size // 64)
        ty = size - th - pad_y * 2
        draw.rectangle((0, ty - pad_y, size, ty + th + pad_y * 2), fill=(0, 0, 0, 160))
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        im.save(dest_path, 'PNG', optimize=True)
        return True
    except Exception as e:
        print(f'[asset-browser] placeholder gen failed for {filename_for_label}: {e}')
        return False


def _make_preview_thumb(source_path: str, dest_path: str) -> bool:
    """256x256 JPEG centre-crop thumbnail. Cached by source mtime."""
    try:
        if (os.path.isfile(dest_path)
                and os.path.getmtime(dest_path) >= os.path.getmtime(source_path)):
            return True
        from PIL import Image
        with Image.open(source_path) as im:
            im = im.convert('RGB')
            w, h = im.size
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            im = im.crop((left, top, left + side, top + side))
            sz = _thumb_size()
            im = im.resize((sz, sz), Image.LANCZOS)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            im.save(dest_path, 'JPEG', quality=_thumb_quality(), optimize=True)
        return True
    except Exception as e:
        print(f'[asset-browser] preview thumb failed for {source_path}: {e}')
        return False


def _copy_full_preview(source_path: str, dest_path: str) -> bool:
    """Copy the original sidecar preview to outputs/_previews/<kind>/<hash>_full.<ext>
    so the SPA lightbox can show it at native resolution (instead of the 256x256
    thumbnail). Cached by source mtime — only re-copies if source is newer.
    """
    try:
        if (os.path.isfile(dest_path)
                and os.path.getmtime(dest_path) >= os.path.getmtime(source_path)):
            return True
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        import shutil
        shutil.copy2(source_path, dest_path)
        return True
    except Exception as e:
        print(f'[asset-browser] full-preview copy failed for {source_path}: {e}')
        return False


def _probe_image_size(path: str) -> tuple:
    """Cheap (header-only) image dimension probe via Pillow. Returns (w, h) or
    (None, None) on failure. Used so the SPA doesn't have to fetch every
    full-res preview just to read its dimensions before opening the lightbox.
    """
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(im.size[0]), int(im.size[1])
    except Exception:
        return None, None


def _resolve_preview(model_rel_filename: str, model_full_path: str, kind: str) -> tuple:
    """Resolve sidecar preview OR placeholder, cache the thumbnail (always) AND
    the full-resolution copy (when sidecar exists), return
    (thumb_rel_path, full_rel_path, full_w, full_h, preview_kind).

    All paths are relative to outputs/, so the SPA can use them directly.
    preview_kind is 'sidecar' | 'placeholder' | 'missing'.

    full_w/full_h are the dimensions of the FULL preview (preview_full) so the
    SPA can lay the lightbox out without an upfront fetch+probe per item.

    For sidecars: full = copy of the original sidecar at native resolution.
    For placeholders: full = the PLACEHOLDER_FULL_SIZE square PNG.
    """
    cache_id = _hash_id(model_rel_filename)
    thumb_dir = _previews_dir(kind)
    thumb_path = os.path.join(thumb_dir, f'{cache_id}.jpg')
    rel_thumb = os.path.relpath(thumb_path, _outputs_root()).replace(os.sep, '/')

    sidecar = _find_sidecar_preview(model_full_path)
    if sidecar and _make_preview_thumb(sidecar, thumb_path):
        ext = os.path.splitext(sidecar)[1].lower() or '.png'
        full_path = os.path.join(thumb_dir, f'{cache_id}_full{ext}')
        if _copy_full_preview(sidecar, full_path):
            rel_full = os.path.relpath(full_path, _outputs_root()).replace(os.sep, '/')
        else:
            rel_full = rel_thumb
            full_path = os.path.join(_outputs_root(), rel_thumb)
        fw, fh = _probe_image_size(full_path)
        return rel_thumb, rel_full, fw, fh, 'sidecar'

    placeholder_dir = os.path.join(_outputs_root(), PREVIEWS_DIR_NAME, PLACEHOLDER_DIR_NAME)
    placeholder_png = os.path.join(placeholder_dir, f'{cache_id}.png')
    if _make_placeholder_png(model_rel_filename, placeholder_png):
        if _make_preview_thumb(placeholder_png, thumb_path):
            rel_full = os.path.relpath(placeholder_png, _outputs_root()).replace(os.sep, '/')
            # Placeholders are always rendered at PLACEHOLDER_FULL_SIZE squares,
            # so no need to pay for a probe.
            return rel_thumb, rel_full, PLACEHOLDER_FULL_SIZE, PLACEHOLDER_FULL_SIZE, 'placeholder'
    return '', '', None, None, 'missing'


# --------------------------------------------------------------------------
# Item builders (one per kind)
# --------------------------------------------------------------------------

def _stat_file(path: str) -> dict:
    try:
        st = os.stat(path)
        return {
            'size_bytes': int(st.st_size),
            'modified_at': datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds'),
        }
    except Exception:
        return {'size_bytes': None, 'modified_at': None}


def _read_cached_triggers(filename: str, kind: str) -> list:
    """Return triggers from local metadata + cached CivitAI (no API). Deduped, local first."""
    try:
        from modules import lora_metadata
        if kind == 'embedding':
            local = lora_metadata.get_embedding_triggers_from_file(filename, [modules.config.path_embeddings])
        else:
            local = lora_metadata.get_lora_triggers_from_file(filename, modules.config.paths_loras)
    except Exception:
        local = {}

    try:
        from modules import civitai_api
        civitai_kind = 'lora' if kind == 'lora' else ('embedding' if kind == 'embedding' else None)
        cached = civitai_api.load_cached_triggers(filename, kind=civitai_kind) if civitai_kind else None
    except Exception:
        cached = None

    merged = []
    seen = set()
    for w in (local or {}).get('trainedWords', []) or []:
        wl = str(w).strip().lower()
        if wl and wl not in seen:
            merged.append(w); seen.add(wl)
    for w in (cached or {}).get('trainedWords', []) or []:
        wl = str(w).strip().lower()
        if wl and wl not in seen:
            merged.append(w); seen.add(wl)
    return merged


def _civitai_url_from_cache(filename: str, kind: str) -> str:
    """Best-effort CivitAI URL from cached metadata. Empty if unknown."""
    try:
        from modules import civitai_api
        if kind == 'checkpoint':
            cached = civitai_api.load_cached_settings(filename)
        else:
            cached = civitai_api.load_cached_triggers(filename, kind=kind)
        if not cached:
            return ''
        # Several possible shapes — try a few keys.
        info = cached.get('model_info') or cached.get('model') or cached
        model_id = info.get('modelId') or info.get('id') or info.get('model_id')
        if model_id:
            return f'https://civitai.com/models/{model_id}'
    except Exception:
        pass
    return ''


def _checkpoint_consensus_from_cache(filename: str) -> dict:
    """Pull sampler/cfg/steps/clip_skip from civitai_cache if present. Empty otherwise."""
    try:
        from modules import civitai_api
        cached = civitai_api.load_cached_settings(filename)
    except Exception:
        cached = None
    if not cached or 'settings' not in cached:
        return {}
    s = cached.get('settings', {}) or {}
    out = {}
    for k_civ, k_out in [('sampler', 'sampler'), ('cfg', 'cfg'), ('steps', 'steps'),
                          ('clip_skip', 'clip_skip'), ('top_resolution', 'top_resolution')]:
        v = s.get(k_civ)
        if v not in (None, ''):
            out[k_out] = v
    base = (cached.get('model_info') or {}).get('baseModel')
    if base:
        out['base_model'] = base
    return out


def _build_lora_item(rel_filename: str, full_path: str) -> dict:
    preview, preview_full, fw, fh, preview_kind = _resolve_preview(rel_filename, full_path, 'loras')
    item = {
        'id': _hash_id(rel_filename, 16),
        'filename': os.path.basename(rel_filename),
        'rel_path': rel_filename.replace(os.sep, '/'),
        'subfolder': os.path.dirname(rel_filename).replace(os.sep, '/') or '.',
        'preview': preview,
        'preview_full': preview_full,
        'preview_width': fw,
        'preview_height': fh,
        'preview_kind': preview_kind,
        'trigger_words': _read_cached_triggers(rel_filename, 'lora'),
        'civitai_url': _civitai_url_from_cache(rel_filename, 'lora'),
    }
    item.update(_stat_file(full_path))
    return item


def _build_checkpoint_item(rel_filename: str, full_path: str) -> dict:
    preview, preview_full, fw, fh, preview_kind = _resolve_preview(rel_filename, full_path, 'checkpoints')
    consensus = _checkpoint_consensus_from_cache(rel_filename)
    item = {
        'id': _hash_id(rel_filename, 16),
        'filename': os.path.basename(rel_filename),
        'rel_path': rel_filename.replace(os.sep, '/'),
        'subfolder': os.path.dirname(rel_filename).replace(os.sep, '/') or '.',
        'preview': preview,
        'preview_full': preview_full,
        'preview_width': fw,
        'preview_height': fh,
        'preview_kind': preview_kind,
        'base_model': consensus.pop('base_model', None),
        'civitai_consensus': consensus or None,
        'civitai_url': _civitai_url_from_cache(rel_filename, 'checkpoint'),
    }
    item.update(_stat_file(full_path))
    return item


def _build_embedding_item(rel_filename: str, full_path: str) -> dict:
    preview, preview_full, fw, fh, preview_kind = _resolve_preview(rel_filename, full_path, 'embeddings')
    base = os.path.splitext(os.path.basename(rel_filename))[0].lower()
    is_negative_hint = any(base.startswith(p) for p in NEGATIVE_PREFIXES)
    item = {
        'id': _hash_id(rel_filename, 16),
        'filename': os.path.basename(rel_filename),
        'rel_path': rel_filename.replace(os.sep, '/'),
        'subfolder': os.path.dirname(rel_filename).replace(os.sep, '/') or '.',
        'preview': preview,
        'preview_full': preview_full,
        'preview_width': fw,
        'preview_height': fh,
        'preview_kind': preview_kind,
        'trigger': os.path.splitext(os.path.basename(rel_filename))[0],
        'is_negative_hint': is_negative_hint,
        'trigger_words': _read_cached_triggers(rel_filename, 'embedding'),
        'civitai_url': _civitai_url_from_cache(rel_filename, 'embedding'),
    }
    item.update(_stat_file(full_path))
    return item


# --------------------------------------------------------------------------
# Scanners
# --------------------------------------------------------------------------

def _resolve_full_path(rel_filename: str, paths) -> str:
    """Wrapper around get_file_from_folder_list that swallows misses."""
    try:
        full = get_file_from_folder_list(rel_filename, paths if isinstance(paths, list) else [paths])
        if full and os.path.isfile(full):
            return full
    except Exception:
        pass
    return ''


def scan_loras() -> list:
    items = []
    for filename in (modules.config.lora_filenames or []):
        full = _resolve_full_path(filename, modules.config.paths_loras)
        if not full:
            continue
        try:
            items.append(_build_lora_item(filename, full))
        except Exception as e:
            print(f'[asset-browser] scan_loras item failed for {filename}: {e}')
    return items


def scan_checkpoints() -> list:
    items = []
    for filename in (modules.config.model_filenames or []):
        full = _resolve_full_path(filename, modules.config.paths_checkpoints)
        if not full:
            continue
        try:
            items.append(_build_checkpoint_item(filename, full))
        except Exception as e:
            print(f'[asset-browser] scan_checkpoints item failed for {filename}: {e}')
    return items


def scan_embeddings() -> list:
    items = []
    for filename in (modules.config.embedding_filenames or []):
        full = _resolve_full_path(filename, [modules.config.path_embeddings])
        if not full:
            continue
        try:
            items.append(_build_embedding_item(filename, full))
        except Exception as e:
            print(f'[asset-browser] scan_embeddings item failed for {filename}: {e}')
    return items


# --------------------------------------------------------------------------
# Top-level orchestration
# --------------------------------------------------------------------------

def _write_manifest(name: str, items: list) -> None:
    os.makedirs(_index_dir(), exist_ok=True)
    payload = {
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'count': len(items),
        'items': items,
    }
    path = os.path.join(_index_dir(), name)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def scan_all_and_write() -> tuple:
    """Top-level entrypoint. Idempotent. Returns (ok, summary_dict)."""
    if not _enabled():
        return False, {'reason': 'asset_browser disabled'}
    summary = {}
    try:
        with _lock:
            loras = scan_loras()
            _write_manifest('loras.json', loras)
            summary['loras'] = len(loras)

            checkpoints = scan_checkpoints()
            _write_manifest('checkpoints.json', checkpoints)
            summary['checkpoints'] = len(checkpoints)

            embeddings = scan_embeddings()
            _write_manifest('embeddings.json', embeddings)
            summary['embeddings'] = len(embeddings)
        print(f'[asset-browser] scan complete: {summary}')
        return True, summary
    except Exception as e:
        print(f'[asset-browser] scan_all_and_write failed: {e}')
        summary['error'] = str(e)
        return False, summary


def maybe_start_boot_scan() -> None:
    if not _index_on_boot_enabled():
        return
    threading.Thread(
        target=scan_all_and_write,
        name='asset-browser-bootscan',
        daemon=True,
    ).start()
    print('[asset-browser] boot scan thread started.')


# --------------------------------------------------------------------------
# CivitAI preview fetcher (custom-8.x — UI-triggered from the SPA lightbox)
# --------------------------------------------------------------------------

_KIND_TO_PATHS = {
    'loras':        lambda: modules.config.paths_loras,
    'checkpoints':  lambda: modules.config.paths_checkpoints,
    'embeddings':   lambda: [modules.config.path_embeddings],
}
_KIND_TO_CIVIT_KIND = {'loras': 'lora', 'checkpoints': 'checkpoint', 'embeddings': 'embedding'}


def fetch_civitai_preview_for(kind: str, rel_filename: str, api_key: str = None) -> dict:
    """Fetch the first CivitAI image for a model with no sidecar preview.
    Saves it as `<model_dir>/<stem>.preview.png` (A1111/ComfyUI convention).

    Returns a dict the SPA can consume:
        {success: bool, message: str, kind: str, rel_path: str}
    On success the model's manifest is rebuilt so the next reload picks
    up the new preview without a full Reindex.
    """
    if kind not in _KIND_TO_PATHS:
        return {'success': False, 'message': f'Unknown kind: {kind!r}',
                'kind': kind, 'rel_path': rel_filename}
    if not _enabled():
        return {'success': False, 'message': 'Asset Browser is disabled.',
                'kind': kind, 'rel_path': rel_filename}

    paths = _KIND_TO_PATHS[kind]()
    full = _resolve_full_path(rel_filename, paths)
    if not full:
        return {'success': False, 'message': f'Model not found on disk: {rel_filename}',
                'kind': kind, 'rel_path': rel_filename}

    # Refuse to overwrite an existing sidecar — that would silently destroy
    # a user's hand-curated preview. The user can delete it first if intended.
    if _find_sidecar_preview(full):
        return {'success': False,
                'message': 'A sidecar preview already exists. Delete it first to re-fetch.',
                'kind': kind, 'rel_path': rel_filename}

    api_key = api_key or getattr(modules.config, 'civitai_api_key', None) or None

    try:
        from modules import civitai_api
        sha = civitai_api._get_full_sha256(full)
        if not sha:
            return {'success': False, 'message': 'Could not hash the model file.',
                    'kind': kind, 'rel_path': rel_filename}
        version = civitai_api.get_model_version_by_hash(sha, api_key=api_key)
        if not version:
            return {'success': False, 'message': 'Not found on CivitAI (hash unknown).',
                    'kind': kind, 'rel_path': rel_filename}
        version_id = version.get('modelVersionId') or version.get('id')
        if not version_id:
            return {'success': False, 'message': 'CivitAI returned no version id.',
                    'kind': kind, 'rel_path': rel_filename}
        images = civitai_api.get_top_images(version_id, api_key=api_key, limit=5)
        if not images:
            return {'success': False, 'message': 'CivitAI has no images for this model.',
                    'kind': kind, 'rel_path': rel_filename}
        # Pick the first usable URL (most-reactions sort).
        img_url = None
        for it in images:
            u = it.get('url') if isinstance(it, dict) else None
            if u:
                img_url = u
                break
        if not img_url:
            return {'success': False, 'message': 'CivitAI returned no usable image URL.',
                    'kind': kind, 'rel_path': rel_filename}
    except Exception as e:
        return {'success': False, 'message': f'CivitAI lookup failed: {e}',
                'kind': kind, 'rel_path': rel_filename}

    # Download → re-encode as PNG to normalise format (CivitAI may serve JPEG/WebP).
    try:
        import urllib.request
        from io import BytesIO
        from PIL import Image
        req = urllib.request.Request(img_url, headers={'User-Agent': 'Fooocus2025/asset-browser'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        im = Image.open(BytesIO(data)).convert('RGB')
        stem = os.path.splitext(os.path.basename(full))[0]
        target_dir = os.path.dirname(full)
        target_path = os.path.join(target_dir, f'{stem}.preview.png')
        im.save(target_path, 'PNG', optimize=True)
        print(f'[asset-browser] fetched CivitAI preview for {rel_filename} -> {target_path}')
    except Exception as e:
        return {'success': False, 'message': f'Download/save failed: {e}',
                'kind': kind, 'rel_path': rel_filename}

    # Invalidate the cached thumbnail / full copy for this model so the next
    # rebuild sees the new sidecar instead of the stale placeholder cache.
    try:
        cache_id = _hash_id(rel_filename)
        for stale in [
            os.path.join(_previews_dir(kind), f'{cache_id}.jpg'),
            os.path.join(_previews_dir(kind), f'{cache_id}_full.png'),
            os.path.join(_previews_dir(kind), f'{cache_id}_full.jpg'),
            os.path.join(_outputs_root(), PREVIEWS_DIR_NAME, PLACEHOLDER_DIR_NAME, f'{cache_id}.png'),
        ]:
            if os.path.isfile(stale):
                try: os.remove(stale)
                except Exception: pass
    except Exception:
        pass

    # Rebuild the manifest for this kind so the SPA picks up preview_full + thumb.
    try:
        rebuilders = {'loras': scan_loras, 'checkpoints': scan_checkpoints, 'embeddings': scan_embeddings}
        items = rebuilders[kind]()
        _write_manifest(f'{kind}.json', items)
    except Exception as e:
        return {'success': True,
                'message': f'Preview saved, but manifest rebuild failed: {e}. Click Reindex to refresh.',
                'kind': kind, 'rel_path': rel_filename}

    return {'success': True,
            'message': f'Preview fetched: {os.path.basename(target_path)}',
            'kind': kind, 'rel_path': rel_filename}
