"""custom-13 — Tag Autocomplete (tags booru + assets locaux).

Backend minimal, appele une seule fois au boot par
`modules/ui_gradio_extensions.py::javascript_html()` et uniquement si
`tag_autocomplete.enabled = true` dans config.txt.

Deux responsabilites :
  1. ensure_tag_sources() : telecharge (premier lancement seulement) les CSV
     de tags danbooru/e621 du projet a1111-sd-webui-tagcomplete dans `tags/`.
     Format CSV : nom,categorie,count,"alias1,alias2".
  2. build_local_assets() : genere `tags/local_assets.json` a chaque boot en
     fusionnant les trigger words LoRA/embeddings (civitai_cache/*.civitai.json),
     les noms d'embeddings et les noms de wildcards.

Les deux fichiers sont ensuite servis au navigateur par la route Gradio
`/file=` (comme les .js), et tout le reste vit cote client dans
`javascript/tag_autocomplete.js`. Aucun endpoint, aucun thread, aucun cout
en generation.
"""
import json
import os
import urllib.request

import modules.config

TAG_SOURCES = {
    'danbooru': 'https://raw.githubusercontent.com/DominikDoom/a1111-sd-webui-tagcomplete/main/tags/danbooru.csv',
    'e621': 'https://raw.githubusercontent.com/DominikDoom/a1111-sd-webui-tagcomplete/main/tags/e621.csv',
}

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)
tags_root = os.path.join(script_path, 'tags')
civitai_cache_dir = os.path.join(script_path, 'civitai_cache')


def _log(msg):
    print(f'[TagAC] {msg}')


def source_csv_path(name):
    return os.path.join(tags_root, f'{name}.csv')


def local_assets_path():
    return os.path.join(tags_root, 'local_assets.json')


def _download(url, dst, timeout=30):
    """Telechargement atomique (tmp + rename) pour ne jamais servir un CSV tronque."""
    tmp = dst + '.tmp'
    req = urllib.request.Request(url, headers={'User-Agent': 'Fooocus2026-TagAC/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, 'wb') as f:
        f.write(r.read())
    os.replace(tmp, dst)


def ensure_tag_sources():
    """Telecharge les CSV manquants. Ne re-telecharge jamais un fichier present
    (suppression manuelle du fichier = maniere officielle de forcer une mise a jour)."""
    os.makedirs(tags_root, exist_ok=True)
    wanted = modules.config.tag_autocomplete_setting('sources') or []
    available = []
    for name in wanted:
        if name not in TAG_SOURCES:
            _log(f'WARNING: source inconnue "{name}" ignoree (choix: {sorted(TAG_SOURCES)})')
            continue
        dst = source_csv_path(name)
        if not os.path.isfile(dst):
            _log(f'Telechargement des tags {name} (premier lancement)...')
            try:
                _download(TAG_SOURCES[name], dst)
                _log(f'OK: {os.path.basename(dst)} ({os.path.getsize(dst) // 1024} Ko)')
            except Exception as e:
                _log(f'WARNING: telechargement {name} impossible ({e}). '
                     f'L\'autocomplete fonctionnera sans cette source.')
                continue
        available.append(name)
    return available


def _stem(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def _read_civitai_triggers(suffix):
    """Liste [(stem, [triggers])] depuis civitai_cache/*.{suffix}.civitai.json."""
    out = []
    if not os.path.isdir(civitai_cache_dir):
        return out
    tail = f'.{suffix}.civitai.json'
    for fn in sorted(os.listdir(civitai_cache_dir)):
        if not fn.endswith(tail):
            continue
        try:
            with open(os.path.join(civitai_cache_dir, fn), 'r', encoding='utf-8') as f:
                data = json.load(f)
            triggers = [t.strip() for t in (data.get('trainedWords') or [])
                        if isinstance(t, str) and t.strip()]
            out.append((fn[:-len(tail)], triggers))
        except Exception:
            continue  # un cache corrompu ne doit jamais bloquer le boot
    return out


def build_local_assets():
    """Genere tags/local_assets.json. Regenere a chaque boot (scan < 100 ms)."""
    os.makedirs(tags_root, exist_ok=True)
    cfg = modules.config.tag_autocomplete_setting
    data = {'loras': [], 'embeddings': [], 'wildcards': []}

    if cfg('suggest_lora_triggers'):
        for stem, triggers in _read_civitai_triggers('lora'):
            if triggers:
                data['loras'].append({'name': stem, 'triggers': triggers})

    if cfg('suggest_embeddings'):
        civitai_emb = dict(_read_civitai_triggers('embedding'))
        for fn in getattr(modules.config, 'embedding_filenames', []):
            stem = _stem(fn)
            data['embeddings'].append({
                'name': stem,
                'triggers': civitai_emb.get(stem, []),
            })

    if cfg('suggest_wildcards'):
        for fn in getattr(modules.config, 'wildcard_filenames', []):
            data['wildcards'].append(_stem(fn))

    tmp = local_assets_path() + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(tmp, local_assets_path())
    _log(f"Assets locaux indexes: {len(data['loras'])} LoRAs, "
         f"{len(data['embeddings'])} embeddings, {len(data['wildcards'])} wildcards")
    return data


def init():
    """Point d'entree unique. Toujours non-bloquant pour le boot en cas d'erreur."""
    sources = []
    try:
        sources = ensure_tag_sources()
    except Exception as e:
        _log(f'WARNING: ensure_tag_sources a echoue: {e}')
    try:
        build_local_assets()
    except Exception as e:
        _log(f'WARNING: build_local_assets a echoue: {e}')
    return sources
