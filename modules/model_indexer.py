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


# --------------------------------------------------------------------------
# File delete (custom-8.7 — Asset Browser lightbox Delete button)
# --------------------------------------------------------------------------
# Sends the model/image file AND its sidecar previews/thumbnails to the OS
# trash (send2trash if available, else hard delete with warning). Then rebuilds
# the appropriate manifest so the SPA refreshes without a full Reindex.
#
# Supported kinds:
#   'loras' | 'checkpoints' | 'embeddings' — rel_filename is the model rel path.
#   'outputs'                              — rel_filename is "<date>/<image>".

def _windows_recycle(filepath: str) -> bool:
    """Move a file to the Windows Recycle Bin via the native VisualBasic API.
    Works without any pip dependency — invokes PowerShell which is always
    present on Win10/11. Returns True on success.
    """
    try:
        import platform, subprocess
        if platform.system() != 'Windows':
            return False
        # Single PowerShell call per file. ~300-500ms overhead each — fine
        # because the user triggers Delete manually, one at a time.
        # Note: -LiteralPath avoids glob expansion on names with [], $, etc.
        script = (
            "Add-Type -AssemblyName Microsoft.VisualBasic;"
            "[Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile("
            f"'{filepath.replace(chr(39), chr(39)*2)}',"
            "'OnlyErrorDialogs','SendToRecycleBin')"
        )
        r = subprocess.run(
            ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception as e:
        print(f'[asset-browser] windows recycle failed for {filepath}: {e}')
        return False


def _to_trash_or_remove(filepath: str, deleted_list: list) -> tuple:
    """Best-effort: send to OS trash if possible, else hard remove.
    Mutates deleted_list. Returns (ok: bool, mode: str).

    Order of preference:
      1. send2trash (cross-platform, if installed)
      2. PowerShell + Microsoft.VisualBasic.FileIO (Windows native, no deps)
      3. os.remove() (permanent — last resort)
    """
    if not os.path.isfile(filepath):
        return False, 'missing'
    # 1. send2trash if installed
    try:
        import send2trash
        try:
            send2trash.send2trash(filepath)
            deleted_list.append(os.path.basename(filepath))
            return True, 'trash'
        except Exception as e:
            print(f'[asset-browser] send2trash failed, falling back: {e}')
    except ImportError:
        pass
    # 2. Windows native recycle (no extra deps)
    if _windows_recycle(filepath):
        deleted_list.append(os.path.basename(filepath))
        return True, 'trash'
    # 3. Hard delete (permanent)
    try:
        os.remove(filepath)
        deleted_list.append(os.path.basename(filepath))
        print(f'[asset-browser] WARNING: hard-deleted {filepath} (recycle unavailable)')
        return True, 'remove'
    except Exception as e:
        print(f'[asset-browser] delete failed for {filepath}: {e}')
        return False, 'error'


def _delete_model_file(kind: str, rel_filename: str) -> dict:
    """Delete a model + all its sidecar previews. Rebuild the kind's manifest."""
    paths = _KIND_TO_PATHS[kind]()
    full = _resolve_full_path(rel_filename, paths)
    if not full:
        return {'success': False,
                'message': f'Model not found on disk: {rel_filename}',
                'kind': kind, 'rel_path': rel_filename}
    deleted = []
    # Main model file
    ok, mode = _to_trash_or_remove(full, deleted)
    if not ok:
        return {'success': False,
                'message': f'Could not delete {os.path.basename(full)} ({mode}).',
                'kind': kind, 'rel_path': rel_filename}
    # All known sidecar previews
    base, _ = os.path.splitext(full)
    for suffix in PREVIEW_SUFFIXES:
        cand = base + suffix
        if os.path.isfile(cand):
            _to_trash_or_remove(cand, deleted)
    # CivitAI sidecar JSON cache (filename.civitai.json convention)
    for sidecar_ext in ('.civitai.json', '.json', '.txt'):
        cand = base + sidecar_ext
        if os.path.isfile(cand):
            _to_trash_or_remove(cand, deleted)
    # Indexer-side cached previews under outputs/_previews/<kind>/
    try:
        cache_id = _hash_id(rel_filename, 16)
        for stale in [
            os.path.join(_previews_dir(kind), f'{cache_id}.jpg'),
            os.path.join(_previews_dir(kind), f'{cache_id}_full.jpg'),
            os.path.join(_outputs_root(), PREVIEWS_DIR_NAME, PLACEHOLDER_DIR_NAME, f'{cache_id}.png'),
        ]:
            if os.path.isfile(stale):
                try: os.remove(stale)  # internal cache: silent remove, not trash
                except Exception: pass
    except Exception:
        pass
    # Rebuild manifest so the SPA refresh stops listing the deleted item.
    try:
        rebuilders = {'loras': scan_loras, 'checkpoints': scan_checkpoints, 'embeddings': scan_embeddings}
        items = rebuilders[kind]()
        _write_manifest(f'{kind}.json', items)
    except Exception as e:
        return {'success': True,
                'message': f'Deleted {len(deleted)} file(s) but manifest rebuild failed: {e}. Click Reindex.',
                'kind': kind, 'rel_path': rel_filename, 'files_deleted': deleted}
    return {'success': True,
            'message': f'Deleted {len(deleted)} file(s) to trash: {", ".join(deleted[:4])}'
                       + (f' (+{len(deleted)-4} more)' if len(deleted) > 4 else ''),
            'kind': kind, 'rel_path': rel_filename, 'files_deleted': deleted}


def _delete_output_image(rel_filename: str) -> dict:
    """rel_filename = '<date>/<image>'. Delete the image + thumb sidecar + update day manifest."""
    # Lazy import — gallery_writer pulls in PIL etc, we don't want it at module-load time.
    try:
        from modules.gallery_writer import _outputs_root as _ow, _thumbnail_path, MANIFEST_FILE
    except Exception as e:
        return {'success': False, 'message': f'gallery_writer import failed: {e}',
                'kind': 'outputs', 'rel_path': rel_filename}
    rel_norm = rel_filename.replace('\\', '/').lstrip('/')
    parts = rel_norm.split('/', 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return {'success': False, 'message': f'Bad outputs rel_path: {rel_filename!r}',
                'kind': 'outputs', 'rel_path': rel_filename}
    date, image_name = parts
    date_dir = os.path.join(_ow(), date)
    image_path = os.path.join(date_dir, image_name)
    if not os.path.isfile(image_path):
        return {'success': False, 'message': f'Image not found: {rel_filename}',
                'kind': 'outputs', 'rel_path': rel_filename}
    deleted = []
    ok, _ = _to_trash_or_remove(image_path, deleted)
    if not ok:
        return {'success': False, 'message': f'Could not delete {image_name}.',
                'kind': 'outputs', 'rel_path': rel_filename}
    # Thumb sidecar
    thumb = _thumbnail_path(image_path)
    if os.path.isfile(thumb):
        _to_trash_or_remove(thumb, deleted)
    # Patch the day's manifest in-place (cheap — just drop the entry).
    manifest_path = os.path.join(date_dir, MANIFEST_FILE)
    try:
        if os.path.isfile(manifest_path):
            with open(manifest_path, encoding='utf-8') as f:
                manifest = json.load(f)
            before = len(manifest.get('images', []))
            manifest['images'] = [im for im in manifest.get('images', []) if im.get('src') != image_name]
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            print(f'[asset-browser] outputs delete: {image_name} (manifest {before} -> {len(manifest["images"])})')
    except Exception as e:
        return {'success': True,
                'message': f'Deleted {image_name} but manifest patch failed: {e}',
                'kind': 'outputs', 'rel_path': rel_filename, 'files_deleted': deleted}
    return {'success': True,
            'message': f'Deleted {len(deleted)} file(s) to trash: {", ".join(deleted)}',
            'kind': 'outputs', 'rel_path': rel_filename, 'files_deleted': deleted}


def delete_file_for(kind: str, rel_filename: str) -> dict:
    """Public entry point — dispatch to model or outputs deleter.

    Always tries the trash first (send2trash). Falls back to hard os.remove
    with a warning if send2trash isn't installed.
    """
    if not _enabled():
        return {'success': False, 'message': 'Asset Browser is disabled.',
                'kind': kind, 'rel_path': rel_filename}
    if kind == 'outputs':
        return _delete_output_image(rel_filename)
    if kind in _KIND_TO_PATHS:
        return _delete_model_file(kind, rel_filename)
    return {'success': False, 'message': f'Unknown kind: {kind!r}',
            'kind': kind, 'rel_path': rel_filename}


# --------------------------------------------------------------------------
# CivitAI update detection (custom-8.11 — Phase 3A)
# --------------------------------------------------------------------------
# Compares the local model's version-id against the latest available on
# CivitAI for the same model-id. Hit on the SPA lightbox open — never global,
# never automatic. Result cached 24h per file to avoid hammering CivitAI when
# the user reopens the same item.

_UPDATE_CHECK_TTL_SECONDS = 86400   # 24h
_UPDATE_CACHE_SUFFIX = '.updatecheck.json'


def _update_cache_path(rel_filename: str, kind: str) -> str:
    """Where we store the per-file update-check cache (next to the triggers cache)."""
    try:
        from modules import civitai_api
        return civitai_api._get_triggers_cache_path(rel_filename, kind=_KIND_TO_CIVIT_KIND.get(kind, kind)) \
            .replace('.triggers.json', _UPDATE_CACHE_SUFFIX)
    except Exception:
        # Fallback: write inside the indexer cache dir.
        return os.path.join(_index_dir(), f'updatecheck_{_hash_id(rel_filename, 16)}.json')


def _load_cached_update(rel_filename: str, kind: str):
    path = _update_cache_path(rel_filename, kind)
    if not os.path.isfile(path):
        return None
    try:
        mtime = os.path.getmtime(path)
        if (datetime.datetime.now().timestamp() - mtime) > _UPDATE_CHECK_TTL_SECONDS:
            return None
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _save_cached_update(rel_filename: str, kind: str, payload: dict) -> None:
    try:
        path = _update_cache_path(rel_filename, kind)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f'[asset-browser] could not write update cache for {rel_filename}: {e}')


def check_update_for(kind: str, rel_filename: str, force_refresh: bool = False) -> dict:
    """Check whether a local model has a newer version on CivitAI.

    Returns dict the SPA can render directly:
        {
          status: 'up_to_date' | 'update_available' | 'not_on_civitai' | 'error',
          message: str,
          local_version_id: int | None,
          local_version_name: str | None,
          latest_version_id: int | None,
          latest_version_name: str | None,
          latest_published_at: str | None,
          latest_size_mb: float | None,
          latest_download_url: str | None,
          model_name: str | None,
          cached_at: float | None,
          kind: str, rel_path: str,
        }
    """
    out = {'kind': kind, 'rel_path': rel_filename,
           'status': 'error', 'message': '',
           'local_version_id': None, 'local_version_name': None,
           'local_base_model': None,                    # 'SDXL 1.0', 'Flux.1 D', 'Pony', etc.
           'latest_version_id': None, 'latest_version_name': None,
           'latest_published_at': None, 'latest_size_mb': None,
           'latest_download_url': None, 'latest_base_model': None,
           'model_name': None,
           'cached_at': None}
    if kind not in _KIND_TO_PATHS:
        out['message'] = f'Unknown kind: {kind!r}'
        return out
    if not _enabled():
        out['message'] = 'Asset Browser is disabled.'
        return out
    # Try cache first (skipped on force_refresh).
    if not force_refresh:
        cached = _load_cached_update(rel_filename, kind)
        if cached:
            cached['cached_at'] = os.path.getmtime(_update_cache_path(rel_filename, kind))
            return cached
    # Resolve full path + hash.
    paths = _KIND_TO_PATHS[kind]()
    full = _resolve_full_path(rel_filename, paths)
    if not full:
        out['status'] = 'error'
        out['message'] = f'Model not found on disk: {rel_filename}'
        return out
    try:
        from modules import civitai_api
        api_key = getattr(modules.config, 'civitai_api_key', None) or None
        sha = civitai_api._get_full_sha256(full)
        if not sha:
            out['message'] = 'Could not hash the model file.'
            return out
        local_version = civitai_api.get_model_version_by_hash(sha, api_key=api_key)
        if not local_version:
            out['status'] = 'not_on_civitai'
            out['message'] = 'Model not on CivitAI (hash unknown).'
            _save_cached_update(rel_filename, kind, out)
            return out
        out['local_version_id'] = local_version.get('modelVersionId')
        out['local_version_name'] = local_version.get('versionName')
        out['local_base_model'] = local_version.get('baseModel')
        out['model_name'] = local_version.get('modelName')
        model_id = local_version.get('modelId')
        if not model_id:
            out['message'] = 'CivitAI returned no modelId.'
            _save_cached_update(rel_filename, kind, out)
            return out
        latest = civitai_api.get_latest_version_for_model(model_id, api_key=api_key)
        if not latest:
            out['message'] = 'CivitAI returned no model details.'
            _save_cached_update(rel_filename, kind, out)
            return out
        out['latest_version_id'] = latest.get('id')
        out['latest_version_name'] = latest.get('name')
        out['latest_published_at'] = latest.get('publishedAt')
        out['latest_download_url'] = latest.get('downloadUrl')
        out['latest_base_model'] = latest.get('baseModel')
        if latest.get('primary_file'):
            kb = latest['primary_file'].get('sizeKB')
            if kb is not None:
                out['latest_size_mb'] = round(float(kb) / 1024.0, 2)
        # Final verdict.
        if out['local_version_id'] and out['latest_version_id'] \
                and out['local_version_id'] == out['latest_version_id']:
            out['status'] = 'up_to_date'
            out['message'] = f'Up to date (v{out["local_version_name"]})'
        else:
            # CivitAI groups all versions of a model under one page, even when the
            # author later migrates from SDXL to Flux / SD1.5 to SDXL / etc.
            # Loading a Flux file in an SDXL pipeline = guaranteed crash, so we
            # MUST surface the architecture switch instead of pretending it's
            # a drop-in update. baseModel comparison is normalized to ignore
            # tiny string differences ('SDXL 1.0' vs 'SDXL 1.0 LCM' both count
            # as SDXL — same UNet, same loader path).
            def _arch_family(s):
                if not s: return ''
                s = str(s).lower().strip()
                for fam in ('flux', 'sd 3', 'sd3', 'pony', 'sdxl', 'sd 1.5', 'sd 1.4', 'svd'):
                    if fam in s: return fam.replace(' ', '')
                return s.split()[0] if s else ''
            local_fam  = _arch_family(out['local_base_model'])
            latest_fam = _arch_family(out['latest_base_model'])
            if local_fam and latest_fam and local_fam != latest_fam:
                out['status'] = 'upgrade_available'
                out['message'] = (
                    f'Architecture change: {out["local_base_model"]} -> {out["latest_base_model"]}. '
                    f'NOT a drop-in update (latest: {out["latest_version_name"]}).'
                )
            else:
                out['status'] = 'update_available'
                out['message'] = (f'Newer version: {out["latest_version_name"]} '
                                  f'(local: {out["local_version_name"]})')
        _save_cached_update(rel_filename, kind, out)
        return out
    except Exception as e:
        out['status'] = 'error'
        out['message'] = f'Check failed: {e}'
        return out


def _backup_path_for_keep(original_full: str) -> str:
    """Pick a free '<stem>.old<ext>' (then .old.2, .old.3…) next to the original."""
    base, ext = os.path.splitext(original_full)
    cand = f'{base}.old{ext}'
    if not os.path.exists(cand):
        return cand
    i = 2
    while True:
        cand = f'{base}.old.{i}{ext}'
        if not os.path.exists(cand):
            return cand
        i += 1
        if i > 99:
            raise RuntimeError(f'Could not find a free .old.<N> slot for {original_full}')


def _new_path_for_keep_as_is(original_full: str) -> str:
    """For backup_mode='keep_as_is' — original stays untouched, the new download
    lands on '<stem>.new<ext>' (then .new.2, .new.3…)."""
    base, ext = os.path.splitext(original_full)
    cand = f'{base}.new{ext}'
    if not os.path.exists(cand):
        return cand
    i = 2
    while True:
        cand = f'{base}.new.{i}{ext}'
        if not os.path.exists(cand):
            return cand
        i += 1
        if i > 99:
            raise RuntimeError(f'Could not find a free .new.<N> slot for {original_full}')


def _stream_download(url: str, dest_path: str, api_key: str = None,
                     expected_size_bytes: int = None) -> tuple:
    """Stream a download to dest_path. Returns (ok: bool, bytes_written: int, err: str).
    Uses 1 MiB chunks; verifies size match when expected_size_bytes is provided.
    """
    try:
        import urllib.request, urllib.error
        req = urllib.request.Request(url, headers={'User-Agent': 'Fooocus2025-AssetBrowser'})
        if api_key:
            req.add_header('Authorization', f'Bearer {api_key}')
        bytes_written = 0
        with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, 'wb') as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                bytes_written += len(chunk)
        if expected_size_bytes is not None:
            # Allow 1% slack — CivitAI's reported sizeKB is sometimes off by a few bytes.
            if abs(bytes_written - expected_size_bytes) > max(1024, expected_size_bytes // 100):
                return False, bytes_written, (
                    f'Size mismatch: got {bytes_written} bytes, expected ~{expected_size_bytes}')
        return True, bytes_written, ''
    except Exception as e:
        return False, 0, str(e)


def apply_update_for(kind: str, rel_filename: str, backup_mode: str = 'trash') -> dict:
    """Download the latest CivitAI version of a model and place it on disk.

    backup_mode:
      'trash'      — old file goes to OS recycle bin, new takes the original name (recoverable default)
      'keep'       — old file renamed to <stem>.old<ext>, new takes the original name
      'keep_as_is' — old file UNTOUCHED at its original name, new downloaded as <stem>.new<ext>

    Returns:
      {success: bool, message: str, kind, rel_path,
       downloaded_size_mb: float, backup_path: str | None,
       new_path: str | None  (only set when keep_as_is)}
    """
    out = {'success': False, 'kind': kind, 'rel_path': rel_filename,
           'message': '', 'downloaded_size_mb': None, 'backup_path': None,
           'new_path': None}
    if kind not in _KIND_TO_PATHS:
        out['message'] = f'Unknown kind: {kind!r}'; return out
    if not _enabled():
        out['message'] = 'Asset Browser is disabled.'; return out
    if backup_mode not in ('trash', 'keep', 'keep_as_is'):
        out['message'] = f'Invalid backup_mode: {backup_mode!r}'; return out

    # 1. Re-fetch update info (force refresh — cache might be 24h stale).
    info = check_update_for(kind, rel_filename, force_refresh=True)
    if info.get('status') != 'update_available':
        out['message'] = f'No update to apply (status={info.get("status")!r}).'; return out
    download_url = info.get('latest_download_url')
    if not download_url:
        out['message'] = 'CivitAI returned no download URL for the latest version.'; return out

    paths = _KIND_TO_PATHS[kind]()
    original_full = _resolve_full_path(rel_filename, paths)
    if not original_full or not os.path.isfile(original_full):
        out['message'] = f'Original file gone: {rel_filename}'; return out

    # 2. Download to a temp sibling file. Same dir = same filesystem = atomic move.
    tmp_path = original_full + '.downloading'
    if os.path.exists(tmp_path):
        try: os.remove(tmp_path)
        except Exception as e:
            out['message'] = f'Could not clean stale download file: {e}'; return out
    api_key = getattr(modules.config, 'civitai_api_key', None) or None
    expected = None
    if info.get('latest_size_mb'):
        expected = int(float(info['latest_size_mb']) * 1024 * 1024)
    print(f'[asset-browser] update: downloading {download_url} -> {tmp_path}')
    ok, written, err = _stream_download(download_url, tmp_path, api_key=api_key,
                                         expected_size_bytes=expected)
    if not ok:
        try: os.remove(tmp_path)
        except Exception: pass
        out['message'] = f'Download failed: {err}'; return out
    out['downloaded_size_mb'] = round(written / (1024 * 1024), 2)

    # 3. Decide where the new file ends up + how to handle the old one.
    final_new_path = original_full   # default for 'trash' and 'keep'
    if backup_mode == 'trash':
        trash_deleted = []
        ok_trash, mode = _to_trash_or_remove(original_full, trash_deleted)
        if not ok_trash:
            try: os.remove(tmp_path)
            except Exception: pass
            out['message'] = f'Could not move old version to trash ({mode}). Update aborted.'
            return out
        out['backup_path'] = f'(trash) {trash_deleted[0] if trash_deleted else os.path.basename(original_full)}'
    elif backup_mode == 'keep':
        try:
            backup_full = _backup_path_for_keep(original_full)
            os.rename(original_full, backup_full)
            out['backup_path'] = backup_full
        except Exception as e:
            try: os.remove(tmp_path)
            except Exception: pass
            out['message'] = f'Could not rename old version: {e}. Update aborted.'
            return out
    else:  # 'keep_as_is'
        # Old file is UNTOUCHED. New file gets a .new (or .new.N) sibling name.
        try:
            final_new_path = _new_path_for_keep_as_is(original_full)
            out['new_path'] = final_new_path
            out['backup_path'] = f'(unchanged) {os.path.basename(original_full)}'
        except Exception as e:
            try: os.remove(tmp_path)
            except Exception: pass
            out['message'] = f'Could not find a free .new slot: {e}. Update aborted.'
            return out

    # 4. Move the downloaded file into its final place.
    try:
        os.rename(tmp_path, final_new_path)
    except Exception as e:
        out['message'] = (f'Download saved as {tmp_path} but rename to {final_new_path} '
                          f'failed: {e}. Old version status: {out["backup_path"]}.')
        return out

    # 5. Invalidate the per-file update-check cache so the SPA shows fresh state.
    # NOTE: in 'keep_as_is' mode we invalidate the OLD path's cache (the user
    # may still want to re-check it later — and the result is now stale because
    # the .new exists alongside).
    try:
        cache_path = _update_cache_path(rel_filename, kind)
        if os.path.isfile(cache_path):
            os.remove(cache_path)
    except Exception:
        pass

    # 6. Invalidate cached civitai metadata (triggers etc.) — only when the
    # original file was actually replaced (modes 'trash' and 'keep'). In
    # 'keep_as_is' the original is intact, so its cached metadata is still valid.
    if backup_mode != 'keep_as_is':
        try:
            from modules import civitai_api
            for cleaner in (lambda: civitai_api._get_cache_path(os.path.basename(rel_filename)),
                            lambda: civitai_api._get_triggers_cache_path(
                                rel_filename, kind=_KIND_TO_CIVIT_KIND.get(kind, kind))):
                try:
                    p = cleaner()
                    if p and os.path.isfile(p):
                        os.remove(p)
                except Exception:
                    pass
        except Exception:
            pass

    # 7. Rebuild manifest so the SPA refresh picks up the new file
    # (and in 'keep_as_is' mode the .new file also appears as a new entry).
    try:
        rebuilders = {'loras': scan_loras, 'checkpoints': scan_checkpoints, 'embeddings': scan_embeddings}
        items = rebuilders[kind]()
        _write_manifest(f'{kind}.json', items)
    except Exception as e:
        out['success'] = True
        out['message'] = (f'Update applied ({out["downloaded_size_mb"]} MB) but '
                          f'manifest rebuild failed: {e}.')
        return out

    out['success'] = True
    if backup_mode == 'keep_as_is':
        out['message'] = (f'New version downloaded as {os.path.basename(final_new_path)} '
                          f'({out["downloaded_size_mb"]} MB). Old file untouched.')
    else:
        out['message'] = (f'Updated to {info.get("latest_version_name")} '
                          f'({out["downloaded_size_mb"]} MB). Old version: {out["backup_path"]}')
    return out
