"""crispz-sdxl: the SDXL-only model lists a family client may rely on.

Fooocus2026 keeps SD 1.5 checkpoints and every LoRA that is not Flux or SD3 in
its dropdowns (custom-8, modules/config.py: _COMPAT_CHECKPOINT holds 'sd',
_detect_lora_arch is a blacklist). For the UI that is fine, the user chooses.
For cli_caps it is not: crispz-dataset-studio picks any name from caps.models
and caps.loras and expects it to run on the SDXL engine. Measured on Mike's
library (2026-10-31): the czp local route listed 154 checkpoints (SD 1.5, SVD,
an x4 upscaler) and about 2500 LoRAs, Flux, Qwen, Z-Image, Wan and Krea
included, when the arch cache was empty.

This module reads the safetensors headers (no weight loading, same reader as
modules.config) and keeps only what is SDXL, or cannot be read (.ckpt, .pt,
.bin, kept with a warning: the family cannot tell). Results are cached by path
and mtime in crispz_arch_cache.json next to the app: the first scan of a
network drive with thousands of files takes a minute, the next ones nothing.
"""
import json
import os
import re

import cz_core

CACHE_NAME = 'crispz_arch_cache.json'

# SDXL UNet LoRA keys name the attention blocks with several transformer blocks:
# input_blocks.4/5 have 2, input_blocks.7/8 and output_blocks.0/1/2 have 10,
# middle_block.1 has 10. SD 1.5 has exactly one (index 0) everywhere, so any
# transformer_blocks index above 0 is SDXL. Separators vary with the trainer
# (kohya uses '_', diffusers '.').
_SDXL_UNET = re.compile(r'(input_blocks|output_blocks|middle_block)[._]\d+[._]\d+[._]'
                        r'transformer_blocks[._]([1-9]\d*)[._]')
_SDXL_DIFFUSERS = re.compile(r'(down_blocks|up_blocks)[._]\d+[._]attentions[._]\d+[._]'
                             r'transformer_blocks[._]([1-9]\d*)[._]')
_SD_UNET = re.compile(r'(input_blocks|output_blocks|middle_block|down_blocks|up_blocks)[._]')


def _cache_path():
    return os.path.join(cz_core.ROOT, CACHE_NAME)


def _load_cache():
    try:
        with open(_cache_path(), encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(cache):
    try:
        with open(_cache_path(), 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=1)
    except OSError as e:
        cz_core.log(f'arch cache not written ({e}): the headers are read again next time')


def read_keys(path):
    """Tensor names of a safetensors file (header only), None when unreadable."""
    from modules.config import _read_safetensors_keys
    return _read_safetensors_keys(path)


def checkpoint_arch(path):
    """'sdxl', 'sd', 'sd3', 'flux', 'svd', 'other' or 'unknown' (no readable header)."""
    from modules.config import _detect_checkpoint_arch
    arch = _detect_checkpoint_arch(path)
    return 'unknown' if arch == 'no_header' else arch


def lora_arch(path):
    """'sdxl', 'sd', 'flux', 'sd3', 'dit' (Qwen, Z-Image, Wan, Krea and the other
    transformer models), 'not_a_lora', 'other' or 'unknown'."""
    keys = read_keys(path)
    if keys is None:
        return 'unknown'
    joined = '\n'.join(keys)
    if any(k.startswith(('model.diffusion_model.', 'first_stage_model.', 'conditioner.')) for k in keys):
        return 'not_a_lora'
    if 'lora_transformer_joint' in joined or 'joint_blocks' in joined:
        return 'sd3'
    if 'double_blocks' in joined or 'single_blocks' in joined:
        return 'flux'
    if 'lora_te2' in joined or 'text_encoder_2' in joined or 'te2_text' in joined:
        return 'sdxl'
    if _SDXL_UNET.search(joined) or _SDXL_DIFFUSERS.search(joined):
        return 'sdxl'
    if _SD_UNET.search(joined) or 'lora_te1' in joined or 'lora_te_' in joined:
        return 'sd'
    if 'transformer_blocks' in joined or 'transformer.' in joined or 'diffusion_model.blocks' in joined:
        return 'dit'
    return 'other'


def _classify(names, folders, detect, kind, cache):
    kept, hidden, changed = [], [], False
    for name in names:
        full = None
        for folder in folders:
            candidate = os.path.join(folder, name)
            if os.path.isfile(candidate):
                full = candidate
                break
        if full is None:
            kept.append(name)  # not on disk any more: the engine will say so
            continue
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            kept.append(name)
            continue
        key = f'{kind}:{full.replace(chr(92), "/")}'
        entry = cache.get(key)
        if entry and entry.get('mtime') == mtime:
            arch = entry['arch']
        else:
            arch = detect(full)
            cache[key] = {'arch': arch, 'mtime': mtime}
            changed = True
        if arch in ('sdxl', 'unknown'):
            kept.append(name)
        else:
            hidden.append((name, arch))
    return kept, hidden, changed


def sdxl_lists(models, loras, model_folders, lora_folders):
    """(models_kept, loras_kept, report). report = {'checkpoints': {arch: count},
    'loras': {arch: count}} of what was hidden, for caps and the console."""
    cache = _load_cache()
    kept_models, hidden_models, c1 = _classify(models, model_folders, checkpoint_arch, 'checkpoint', cache)
    kept_loras, hidden_loras, c2 = _classify(loras, lora_folders, lora_arch, 'lora', cache)
    if c1 or c2:
        _save_cache(cache)
    report = {}
    for label, hidden in (('checkpoints', hidden_models), ('loras', hidden_loras)):
        counts = {}
        for _, arch in hidden:
            counts[arch] = counts.get(arch, 0) + 1
        if counts:
            report[label] = counts
    if report:
        cz_core.log('caps hide what is not SDXL: ' + '; '.join(
            f'{label} ' + ', '.join(f'{n} {arch}' for arch, n in sorted(c.items())) for label, c in report.items()))
    return kept_models, kept_loras, report
