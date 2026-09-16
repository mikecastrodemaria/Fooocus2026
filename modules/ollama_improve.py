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
    "terms; keep the format of the input (a tag list stays a tag list, prose stays prose, "
    "see INPUT FORMAT when given); do not pad it with generic quality filler (masterpiece, "
    "best quality, 8k). Output ONLY the improved prompt, on one line, no preamble, no "
    "quotes, no explanation.\n\nPROMPT: {prompt}")

# custom-34: the input format is detected in code (small models guess it badly, above all
# with braces or wildcards in the text) and stated to the model. Config ollama_improve.format:
# 'auto' (default), 'tags', 'prose' or 'off'.
FORMAT_NOTES = {
    'tags': ("INPUT FORMAT: comma-separated tags. Answer in the SAME format: one line of "
             "comma-separated tags or short phrases (2 to 5 words each), most specific first; "
             "no full sentences, no narrative, no bullet points."),
    'prose': ("INPUT FORMAT: prose. Answer in the SAME format: one flowing paragraph of "
              "natural sentences, no comma-separated tag list, no bullet points, no "
              "keyword dump at the end."),
}
_FORMAT_CHOICES = ('auto', 'tags', 'prose', 'off')
_SENTENCE_END_RE = re.compile(r'[.!?](?:\s|$)')
_STRIP_SYNTAX_RE = re.compile(r'\{[^{}]*\}|__[\w-]+__|<lora:[^>]*>|\([^()]*:\s*[\d.]+\)')


def detect_format(text):
    """'tags' or 'prose' for the positive prompt, from the text alone.

    Dynamic syntax ({a|b}, __wildcards__, <lora:...>, (word:1.2)) is blanked first so it
    never tips the balance. Sentence punctuation inside the text means prose; otherwise
    the mean length of the comma-separated fragments decides: up to 4 words a fragment is
    a tag list, longer fragments read as prose. A short single phrase ("a fox") is tags.
    """
    cleaned = _STRIP_SYNTAX_RE.sub(' x ', text or '').strip()
    if not cleaned:
        return 'tags'
    inner = cleaned.rstrip('.!? ')
    if _SENTENCE_END_RE.search(inner):
        return 'prose'
    fragments = [f.strip() for f in cleaned.split(',') if f.strip()]
    if not fragments:
        return 'tags'
    mean_words = sum(len(f.split()) for f in fragments) / len(fragments)
    return 'tags' if mean_words <= 4 else 'prose'


def _format_note(text):
    """The INPUT FORMAT block for `text`, or '' (kind negative, config off, empty text)."""
    mode = str(_setting('format', 'auto')).strip().lower()
    if mode not in _FORMAT_CHOICES:
        mode = 'auto'
    if mode == 'off' or not (text or '').strip():
        return ''
    fmt = detect_format(text) if mode == 'auto' else mode
    return FORMAT_NOTES[fmt]


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

# custom-31: what Improve negative starts from when the negative prompt is empty. A plain
# SDXL baseline, overridable with ollama_improve.default_negative in config.txt.
DEFAULT_NEGATIVE = (
    "lowres, worst quality, low quality, jpeg artifacts, blurry, out of focus, "
    "bad anatomy, bad proportions, bad hands, missing fingers, extra digits, fewer digits, "
    "extra limbs, deformed, disfigured, mutated, cropped, cut off, "
    "text, watermark, signature, logo, username")

# custom-31: user directives, appended to the instruction for one call.
DIRECTIVES_HEAD = (
    "USER DIRECTIVES for this rewrite (apply them on top of the rules above; when they "
    "conflict with the rules above, the directives win):\n")


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


def _insert_before_label(tpl, block):
    """Insere `block` juste avant le libelle final PROMPT: / NEGATIVE PROMPT: (sinon a la fin)."""
    last = None
    for last in _LABEL_RE.finditer(tpl):
        pass
    if last is None:                        # custom instruction without the label: append
        return tpl + '\n\n' + block
    return tpl[:last.start()] + '\n\n' + block + tpl[last.start():]


def _instruction(kind, text='', directives=None):
    """Consigne pour ce `kind`; avec la note de syntaxe dynamique si `text` en utilise
    (custom-29) et les directives de l'utilisateur s'il en a donne (custom-31)."""
    if kind == 'negative':
        tpl = _setting('negative_instruction', IMPROVE_NEGATIVE)
    else:
        tpl = _setting('positive_instruction', IMPROVE_POSITIVE)
        note = _format_note(text)              # custom-34 : tags ou prose, comme l'entree
        if note:
            tpl = _insert_before_label(tpl, note)
    if uses_dynamic_syntax(text):
        tpl = _insert_before_label(tpl, SYNTAX_NOTE)
    directives = (directives or '').strip()
    if directives:
        tpl = _insert_before_label(tpl, DIRECTIVES_HEAD + directives)
    return tpl


def default_negative():
    """Le negatif de depart quand la case est vide (config ollama_improve.default_negative)."""
    return str(_setting('default_negative', DEFAULT_NEGATIVE)).strip() or DEFAULT_NEGATIVE


def list_models(base=None):
    """Tous les modeles Ollama installes (le rewrite de texte n'exige pas la vision)."""
    tags = _http('/api/tags', base=base, timeout=5).get('models') or []
    return [m.get('name') for m in tags if m.get('name')]


def improve(text, kind='positive', model=None, base=None, timeout=None, temperature=None,
            directives=None):
    """Reecrit `text` (kind 'positive' ou 'negative'). Renvoie (texte ameliore, modele utilise).
    `directives` : consignes libres de l'utilisateur pour cet appel (custom-31).
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
        'prompt': _instruction(kind, text, directives).replace('{prompt}', text),
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
