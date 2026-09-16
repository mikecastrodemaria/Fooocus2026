"""custom-25 — Describe via un modele vision Ollama : un prompt redige qui reconstruit l'image.

Les deux methodes d'origine de l'onglet Describe (BLIP "Photograph", WD14 "Art/Anime")
rendent une legende courte ou une liste de tags. Ce module demande a un modele vision
local (Ollama) un paragraphe dans un style choisi : prompt en prose, tags, photo
technique, art & style, composition, fiche personnage, texte & typographie.

Porte de crispz-klein 1.36 (cz_core.DESCRIBE_STYLES + cz_ollama). Mesure faite la-bas
(2026-09-11/12, Agents-A1-4B et muse-glimmer, trois images de prompt connu, chaque
description regeneree a seed fixe) : sans le medium, un portrait au crayon revenait en
photo ; demander aussi l'epoque fait passer la fidelite du portrait de 0,54 a 0,65-0,66.
Les consignes sont reprises telles quelles, en anglais : c'est la langue des modeles.

Import-safe : bibliotheque standard (+ PIL/numpy pour l'image). Les reglages sont lus
dans modules.config seulement s'il est deja charge (son import a des effets de bord).
"""
import base64
import io
import json
import re
import sys
import urllib.error
import urllib.request

DEFAULT_ENDPOINT = 'http://localhost:11434'

_DESCRIBE_RULES = (
    "Describe only what is present: never mention what is absent. State every detail as a "
    "fact: no \"appears\", \"seems\", \"likely\", \"possibly\", \"as if\". No filler "
    "(masterpiece, best quality, 8k, stunning, beautiful). Do not start with \"This image\". ")
_DESCRIBE_TEXT_RULE = (
    "Quote text only when it is a main element of the image (a sign, a title or a label in "
    "the foreground): give it once, in reading order, as a single string in double quotes, "
    "exactly as written; skip small or background text entirely. ")
SHORT_CAPTION_STYLE = 'Short caption'
DESCRIBE_STYLES = {
    "Prompt (prose)": (
        "Describe this image as a text-to-image prompt, in one flowing paragraph of about "
        "{words} words. Begin with the medium and style (for example: black-and-white ink "
        "illustration, candid photograph, 3D render, oil painting). Then: main subject(s) "
        "(count, age range, build, face, expression, hair); clothing and accessories "
        "(materials, colors, fit); pose and action; setting from foreground to background, "
        "with positions (left, right, center); camera (shot size, angle, lens, focus); "
        "lighting (sources, direction, softness, color temperature); color palette with "
        "precise color names; time of day, weather and era when they are identifiable; mood. "
        + _DESCRIBE_TEXT_RULE + _DESCRIBE_RULES + "Output only the paragraph."),
    "Prompt (tags)": (
        "Describe this image as a text-to-image prompt made of about {words} words of "
        "comma-separated visual tags, most important first: medium and style, subject, "
        "clothing, pose, setting, camera, lighting, colors, mood. "
        + _DESCRIBE_TEXT_RULE + _DESCRIBE_RULES + "Output only the tags."),
    "Photo (technical)": (
        "Describe this photograph as a text-to-image prompt, in one paragraph of about "
        "{words} words, for a photographer who must reproduce it. Begin with the kind of "
        "photograph (studio portrait, street, product, landscape...). Then: subject and pose; "
        "shot size and camera angle; lens focal length and aperture, depth of field and what "
        "is in focus; lighting setup (key, fill and rim lights, their direction, softness and "
        "color temperature); color grading, contrast, film grain or noise; setting and "
        "background. " + _DESCRIBE_TEXT_RULE + _DESCRIBE_RULES + "Output only the paragraph."),
    "Art & style": (
        "Describe this image as a text-to-image prompt, in one paragraph of about {words} "
        "words, focused on how it is made. Begin with the medium and technique (ink, pencil, "
        "watercolor, oil, digital painting, 3D render, pixel art...). Then: line work and "
        "brush strokes, shading and rendering, color palette with precise color names, level "
        "of detail, art movement or genre, composition; then the subject in a few words. "
        + _DESCRIBE_TEXT_RULE + _DESCRIBE_RULES + "Output only the paragraph."),
    "Composition & layout": (
        "Describe this image as a text-to-image prompt, in one paragraph of about {words} "
        "words, so that the same layout can be rebuilt. Begin with the medium and style. "
        "Then place every element: foreground, middle ground and background; left, center "
        "and right; relative sizes and distances; where the horizon and the vanishing point "
        "sit; framing, camera height and angle; empty space. "
        + _DESCRIBE_TEXT_RULE + _DESCRIBE_RULES + "Output only the paragraph."),
    "Character sheet": (
        "Describe the main character of this image as a text-to-image prompt, in one "
        "paragraph of about {words} words, so that the same character can be drawn again. "
        "Begin with the medium and style. Then: age range, build and height, face shape, "
        "eyes, nose, lips, skin tone, hair (color, length, texture, style); outfit from the "
        "inner layer to the outer one with materials, colors and fit; accessories; "
        "distinguishing marks; pose and expression. Keep the setting to one short sentence. "
        + _DESCRIBE_RULES + "Output only the paragraph."),
    "Text & typography": (
        "Describe this image as a text-to-image prompt, in one paragraph of about {words} "
        "words, for an image whose text matters. Begin with the medium and style. Quote every "
        "legible line of text exactly, in reading order, in double quotes; for each, give its "
        "place, size, font style (serif, sans-serif, script, hand-lettered...), color and "
        "material. Then describe the support (sign, poster, screen, label...) and the "
        "setting. Never guess blurry or partial text. " + _DESCRIBE_RULES
        + "Output only the paragraph."),
    "Dataset paragraph": (
        "Describe this image in one detailed paragraph of about {words} words: subjects and "
        "characters, objects, setting, era if identifiable, medium and technique (photo, "
        "painting, 3D render, illustration...), visual style and mood. Use concrete visual "
        "terms. End with the aspect ratio and orientation. " + _DESCRIBE_RULES
        + "Output only the paragraph."),
    SHORT_CAPTION_STYLE: (
        "Describe this image in one short sentence of at most {words} words: the medium, the "
        "main subject and the setting. " + _DESCRIBE_RULES + "Output only the sentence."),
}
DESCRIBE_LENGTHS = {'Short': 60, 'Medium': 120, 'Long': 180, 'Very long': 300}
DEFAULT_STYLE, DEFAULT_LENGTH = 'Prompt (prose)', 'Long'

# Noms clairement multimodaux, repli quand /api/show ne donne pas les capacites.
_VISION_NAME = ('llava', '-vl', 'vl:', 'moondream', 'minicpm-v', 'bakllava',
                'llama3.2-vision', 'llama-3.2-vision', 'gemma3', 'qwen2.5vl')


class OllamaError(RuntimeError):
    """Message lisible pour l'utilisateur (Ollama arrete, pas de modele vision...)."""


def _setting(key, default):
    cfg = sys.modules.get('modules.config')
    getter = getattr(cfg, 'ollama_describe_setting', None) if cfg is not None else None
    if getter is None:
        return default
    try:
        value = getter(key, default)
        return default if value in (None, '') else value
    except Exception:
        return default


def endpoint():
    return str(_setting('endpoint', DEFAULT_ENDPOINT)).rstrip('/')


def describe_instruction(style=None, length=None):
    """Consigne envoyee au modele pour ce style et cette longueur (defauts sinon)."""
    tpl = DESCRIBE_STYLES.get(style) or DESCRIBE_STYLES[DEFAULT_STYLE]
    words = 25 if style == SHORT_CAPTION_STYLE else DESCRIBE_LENGTHS.get(
        length, DESCRIBE_LENGTHS[DEFAULT_LENGTH])
    return tpl.replace('{words}', str(words))


# Balises de raisonnement des modeles "thinking" (non-greedy, DOTALL).
_THINK_RE = re.compile(r'<\s*(think|thinking|reasoning)\s*>.*?<\s*/\s*\1\s*>', re.IGNORECASE | re.DOTALL)
_THINK_OPEN_RE = re.compile(r'^\s*<\s*(think|thinking|reasoning)\s*>', re.IGNORECASE)


def strip_thinking(text):
    """Retire le monologue interne d'un modele de raisonnement : bloc ferme, bloc ouvert
    jamais referme (il ne reste que du raisonnement), fermeture orpheline."""
    t = _THINK_RE.sub('', text or '')
    if _THINK_OPEN_RE.match(t):
        return ''
    m = re.search(r'<\s*/\s*(think|thinking|reasoning)\s*>', t, re.IGNORECASE)
    if m:
        t = t[m.end():]
    return t.strip()


# Malgre la consigne, un petit modele ecrit encore "No text is visible." ou "appears to be" :
# une absence enoncee peut faire apparaitre la chose dans l'image, une hesitation ne dit rien.
_ABSENCE_RE = re.compile(r"(?i)\b(?:no|without any)\s+(?:visible\s+|other\s+|legible\s+)?"
                         r"(?:text|people|person|one|words|writing|signage|figures|humans)\b"
                         r"|\bnot visible\b|\b(?:is|are) absent\b")


def clean_description(text):
    """Retire les phrases qui enoncent une absence et les tournures d'hesitation. Une
    reponse d'une seule phrase (liste de tags) n'est jamais videe."""
    t = (text or '').strip()
    kept = [s for s in re.split(r'(?<=[.!?])\s+', t) if not _ABSENCE_RE.search(s)]
    out = ' '.join(kept) if kept else t
    out = re.sub(r'(?i)\b(?:appears|seems) to be\b', 'is', out)
    out = re.sub(r'(?i)\b(?:appear|seem) to be\b', 'are', out)
    out = re.sub(r'(?i),?\s*\b(?:likely|possibly|probably|perhaps)\b,?', '', out)
    return re.sub(r'\s{2,}', ' ', out).replace(' ,', ',').replace(' .', '.').strip()


def image_to_b64_jpeg(image, max_side=1024, quality=90):
    """numpy HWC ou PIL -> JPEG base64, cote max borne (les modeles vision n'y gagnent rien
    au-dela et la requete reste legere)."""
    from PIL import Image
    if not isinstance(image, Image.Image):
        import numpy as np
        image = Image.fromarray(np.asarray(image).astype('uint8'))
    img = image.convert('RGB')
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return base64.b64encode(buf.getvalue()).decode('ascii')


# custom-35 : Ollama est local ou sur le LAN, jamais derriere un proxy HTTP. L'ouvreur par
# defaut d'urllib obeit a HTTP_PROXY / HTTPS_PROXY (Pinokio, reseaux d'entreprise) et la
# requete vers localhost partait alors vers le proxy et expirait ("timed out").
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http(path, payload=None, base=None, timeout=8):
    b = (base or endpoint()).rstrip('/')

    def _call(pl):
        data = json.dumps(pl).encode('utf-8') if pl is not None else None
        req = urllib.request.Request(b + path, data=data,
                                     headers={'Content-Type': 'application/json'} if data else {})
        with _OPENER.open(req, timeout=timeout) as r:
            return json.loads(r.read().decode('utf-8'))

    try:
        try:
            return _call(payload)
        except urllib.error.HTTPError as e:
            # un modele sans raisonnement refuse `think` (400) : on rejoue sans le champ
            if e.code == 400 and isinstance(payload, dict) and 'think' in payload:
                return _call({k: v for k, v in payload.items() if k != 'think'})
            raise
    except urllib.error.HTTPError as e:
        detail = ''
        try:
            detail = json.loads(e.read().decode('utf-8', 'replace')).get('error', '')
        except Exception:
            pass
        if e.code == 404 and isinstance(payload, dict) and payload.get('model'):
            raise OllamaError(f'model "{payload["model"]}" not found in Ollama '
                              f'(ollama pull {payload["model"]})')
        raise OllamaError(f'Ollama answered HTTP {e.code} {detail or e.reason}')
    except (urllib.error.URLError, OSError) as e:
        raise OllamaError(f'Ollama unreachable at {b} ({getattr(e, "reason", e)}). Start Ollama '
                          f'or set ollama_describe.endpoint in config.txt.')


def list_vision_models(base=None):
    """Modeles Ollama capables de vision, d'apres la capacite 'vision' de /api/show ;
    repli sur un nom clairement multimodal si /api/show echoue. Les plus fiables d'abord."""
    names = [m.get('name') for m in (_http('/api/tags', base=base, timeout=5).get('models') or [])
             if m.get('name')]
    vision = []
    for n in names:
        try:
            info = _http('/api/show', {'model': n}, base=base, timeout=8)
            caps = [c.lower() for c in (info.get('capabilities') or [])]
            if 'vision' in caps or (not info.get('capabilities')
                                    and any(k in n.lower() for k in _VISION_NAME)):
                vision.append(n)
        except OllamaError:
            if any(k in n.lower() for k in _VISION_NAME):
                vision.append(n)
    vision.sort(key=lambda n: 0 if any(k in n.lower() for k in _VISION_NAME) else 1)
    return vision


def describe(image, model=None, style=None, length=None, base=None, timeout=None, temperature=None):
    """Decrit l'image dans le style choisi. Renvoie (texte nettoye, modele utilise).
    Leve OllamaError avec un message actionnable."""
    model = model or _setting('model', '')
    if not model:
        found = list_vision_models(base)
        if not found:
            raise OllamaError('no vision model in Ollama: pull one first (for example '
                              '"ollama pull qwen2.5vl:7b" or "ollama pull llava")')
        model = found[0]
    payload = {
        'model': model,
        'prompt': describe_instruction(style, length),
        'images': [image_to_b64_jpeg(image)],
        'stream': False,
        'think': False,
        'keep_alive': _setting('keep_alive', '5m'),
        'options': {'temperature': float(temperature if temperature is not None
                                         else _setting('temperature', 0.3))},
    }
    out = _http('/api/generate', payload, base=base, timeout=int(timeout or _setting('timeout', 180)))
    text = clean_description(strip_thinking(out.get('response')))
    if not text:
        raise OllamaError(f'"{model}" returned an empty description (a reasoning model '
                          'may have spent its whole answer thinking)')
    return text, model
