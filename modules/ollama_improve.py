"""custom-28 — Improve a prompt via a text Ollama model: rewrite it richer, same intent.

Companion to custom-25 (Describe). Two buttons, next to the positive and the negative
prompt, send the current text to a local text LLM (Ollama) and replace it with a better
version: the positive prompt is made more vivid and detailed while keeping its subject and
intent; the negative prompt is expanded and tidied into a fuller comma-separated list of
defects to avoid, keeping every term already present.

Reuses the Describe module's Ollama transport (`_http`, `endpoint`, `OllamaError`,
`strip_thinking`) so both features share one server and one endpoint setting. Import-safe:
standard library only. Settings are read from modules.config only if it is already loaded
(same guard as ollama_describe), so importing this module has no side effect.
"""
import re
import sys

from modules.ollama_describe import _http, endpoint, OllamaError, strip_thinking  # shared transport
from modules.prompt_variants import uses_dynamic_syntax

# Consignes en anglais (langue des modeles), reprises de l'esprit de cz_core.IMPROVE_INSTRUCTION.
IMPROVE_POSITIVE = (
    "You are an expert text-to-image prompt writer. Rewrite the following prompt to be more "
    "vivid and detailed while keeping the SAME subject and intent. Prefer concrete visual "
    "terms; keep it comma-separated where that reads naturally; do not pad it with generic "
    "quality filler (masterpiece, best quality, 8k). Output ONLY the improved prompt, on one "
    "line, no preamble, no quotes, no explanation.\n\nPROMPT: {prompt}")
IMPROVE_NEGATIVE = (
    "You are an expert text-to-image prompt writer. The following is a NEGATIVE prompt: a "
    "comma-separated list of things that must NOT appear in the image. Expand and tidy it "
    "into a fuller comma-separated list of common defects and unwanted elements to avoid "
    "(anatomy errors, artifacts, low quality, watermarks, text...), KEEPING every term "
    "already present and removing duplicates. Output ONLY the negative prompt, on one line, "
    "no preamble, no quotes, no explanation.\n\nNEGATIVE PROMPT: {prompt}")

# custom-29: added to the instruction only when the text uses the dynamic syntax, so the
# model keeps {a|b|c} groups and __wildcard__ placeholders instead of expanding them.
SYNTAX_NOTE = (
    "The prompt uses Fooocus dynamic syntax that is resolved later, at generation time, "
    "and must be kept EXACTLY as written: {a|b|c} is a variant group (one option is "
    "picked per image), {2$$a|b|c} picks two options, __name__ is a wildcard file "
    "placeholder. Keep every group and placeholder verbatim: do not expand, reorder, "
    "merge, translate or drop them. You may improve the text around them and the "
    "wording inside each option, as long as the braces, the | separators and the "
    "__names__ stay intact.")

_LABEL_RE = re.compile(r'\n\n(?:NEGATIVE )?PROMPT:')


def _setting(key, default):
    """Lit modules.config.ollama_improve_setting sans importer modules.config (effets de bord)."""
    cfg = sys.modules.get('modules.config')
    getter = getattr(cfg, 'ollama_improve_setting', None) if cfg is not None else None
    if getter is None:
        return default
    try:
        value = getter(key, default)
        return default if value in (None, '') else value
    except Exception:
        return default


def _instruction(kind, text=''):
    """Consigne pour ce `kind`; avec la note de syntaxe dynamique si `text` en utilise."""
    if kind == 'negative':
        tpl = _setting('negative_instruction', IMPROVE_NEGATIVE)
    else:
        tpl = _setting('positive_instruction', IMPROVE_POSITIVE)
    if not uses_dynamic_syntax(text):
        return tpl
    last = None
    for last in _LABEL_RE.finditer(tpl):
        pass
    if last is None:                        # custom instruction without the label: append
        return tpl + '\n\n' + SYNTAX_NOTE
    return tpl[:last.start()] + '\n\n' + SYNTAX_NOTE + tpl[last.start():]


def list_models(base=None):
    """Tous les modeles Ollama installes (le rewrite de texte n'exige pas la vision)."""
    tags = _http('/api/tags', base=base, timeout=5).get('models') or []
    return [m.get('name') for m in tags if m.get('name')]


def improve(text, kind='positive', model=None, base=None, timeout=None, temperature=None):
    """Reecrit `text` (kind 'positive' ou 'negative'). Renvoie (texte ameliore, modele utilise).
    Leve OllamaError avec un message actionnable (Ollama arrete, pas de modele, vide...)."""
    text = (text or '').strip()
    if not text:
        raise OllamaError('nothing to improve: the prompt is empty')
    model = model or _setting('model', '')
    if not model:
        found = list_models(base)
        if not found:
            raise OllamaError('no model in Ollama: pull one first '
                              '(for example "ollama pull llama3.1:8b")')
        model = found[0]
    payload = {
        'model': model,
        'prompt': _instruction(kind, text).replace('{prompt}', text),
        'stream': False,
        'think': False,
        'keep_alive': _setting('keep_alive', '5m'),
        'options': {'temperature': float(temperature if temperature is not None
                                         else _setting('temperature', 0.7))},
    }
    out = _http('/api/generate', payload, base=base,
                timeout=int(timeout or _setting('timeout', 120)))
    result = strip_thinking(out.get('response') or '').strip().strip('"').strip()
    if not result:
        raise OllamaError(f'"{model}" returned an empty result (a reasoning model may have '
                          'spent its whole answer thinking)')
    if uses_dynamic_syntax(text) and not uses_dynamic_syntax(result):
        print(f'[Improve] "{model}" dropped the {{a|b|c}} / __wildcard__ syntax from the prompt; '
              'check the result before generating.')
    return result, model
