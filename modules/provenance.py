"""custom-23 — Provenance IA des images generees (EU AI Act, article 50).

L'article 50(2) demande que les contenus generes par IA soient marques dans un format
lisible par machine. Fooocus n'ecrivait rien de tel : les parametres de generation
(quand ils sont sauves) decrivent le COMMENT, pas le fait qu'une IA a produit l'image,
et ils sont absents si la sauvegarde des metadonnees est coupee.

Ce module ecrit, sur CHAQUE image enregistree :
  - un paquet XMP avec la propriete IPTC standard
    Iptc4xmpExt:DigitalSourceType = trainedAlgorithmicMedia (+ xmp:CreatorTool) :
      PNG  -> chunk iTXt 'XML:com.adobe.xmp' (+ un tEXt lisible 'ai_provenance') ;
      JPEG -> segment APP1 XMP insere apres JFIF / EXIF (Pillow 10 ne sait pas l'ecrire) ;
      WEBP -> chunk XMP natif de Pillow.
    Aucun prompt, aucun parametre : la declaration IA ne revele rien de la creation.
  - en option, un filigrane invisible TrustMark (Adobe, open source), si le paquet
    `trustmark` est installe et provenance.watermark est actif. Payload ~9 caracteres.

Lecture : describe() pour l'onglet Metadata (XMP, et C2PA si c2pa-python est installe).
L'absence de marque ne prouve rien : on n'affiche jamais "authentique" ni "pas IA".

Porte de crispz-studio (cz_provenance.py : TrustMark + lecture C2PA), complete par
l'ecriture de la declaration IPTC que crispz ne faisait pas.
"""
import importlib.util
import os
import re
import sys

DIGITAL_SOURCE_TYPE = 'http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia'
PNG_XMP_KEY = 'XML:com.adobe.xmp'
PNG_TEXT_KEY = 'ai_provenance'
JPEG_XMP_HEADER = b'http://ns.adobe.com/xap/1.0/\x00'
WM_MAX_CHARS = 9  # TrustMark Q + ECC : ~68 bits utiles
NOT_A_PROOF = 'Absence of marks proves nothing: it never means "not AI" or "authentic".'

_warned = set()
_TM = None


def _setting(key, default):
    """Reglage provenance.<key>. Lit modules.config seulement s'il est DEJA charge : son
    import analyse sys.argv (args_manager) et ecrit config.txt, effets de bord hors de
    question depuis un module de sauvegarde ou un test."""
    cfg = sys.modules.get('modules.config')
    getter = getattr(cfg, 'provenance_setting', None) if cfg is not None else None
    if getter is None:
        return default
    try:
        return getter(key, default)
    except Exception:
        return default


def _warn_once(key, msg):
    if key not in _warned:
        _warned.add(key)
        print(msg)


def enabled():
    return bool(_setting('enabled', True))


def generator_name():
    try:
        import fooocus_version
        return f'Fooocus2026 {fooocus_version.version}'
    except Exception:
        return 'Fooocus2026'


def _xml_escape(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;')
            .replace('>', '&gt;').replace('"', '&quot;'))


def xmp_packet(generator=None):
    """Paquet XMP minimal : declaration IPTC + outil createur. Pas de prompt."""
    gen = _xml_escape(generator or generator_name())
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about=""'
        ' xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"'
        ' xmlns:xmp="http://ns.adobe.com/xap/1.0/"'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
        f' Iptc4xmpExt:DigitalSourceType="{DIGITAL_SOURCE_TYPE}"'
        f' xmp:CreatorTool="{gen}">'
        '<dc:description><rdf:Alt><rdf:li xml:lang="x-default">AI-generated image</rdf:li>'
        '</rdf:Alt></dc:description>'
        '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="w"?>')


# ------------------------------------------------------------------- ecriture

def add_to_pnginfo(pnginfo=None, generator=None):
    """Ajoute la declaration a un PngInfo (en cree un si None) et le renvoie."""
    from PIL.PngImagePlugin import PngInfo
    info = pnginfo if pnginfo is not None else PngInfo()
    gen = generator or generator_name()
    info.add_itxt(PNG_XMP_KEY, xmp_packet(gen), zip=False)
    info.add_text(PNG_TEXT_KEY, f'AI-generated image (IPTC DigitalSourceType '
                                f'trainedAlgorithmicMedia), generator: {gen}')
    return info


def webp_save_kwargs(generator=None):
    return {'xmp': xmp_packet(generator).encode('utf-8')}


def inject_jpeg_xmp(path, generator=None):
    """Insere un segment APP1 XMP dans un JPEG deja ecrit, apres SOI et les segments
    APP0 (JFIF) / APP1 (EXIF) existants. Idempotent. Renvoie True si la declaration
    est presente a la fin. Ecriture atomique ; ne leve jamais."""
    try:
        with open(path, 'rb') as f:
            data = f.read()
        if data[:2] != b'\xff\xd8':
            return False
        xmp = JPEG_XMP_HEADER + xmp_packet(generator).encode('utf-8')
        if len(xmp) + 2 > 0xFFFF:
            return False
        pos = 2
        while pos + 4 <= len(data) and data[pos] == 0xFF and data[pos + 1] in (0xE0, 0xE1):
            seg_len = int.from_bytes(data[pos + 2:pos + 4], 'big')
            if data[pos + 1] == 0xE1 and data[pos + 4:pos + 4 + len(JPEG_XMP_HEADER)] == JPEG_XMP_HEADER:
                return True
            pos += 2 + seg_len
        segment = b'\xff\xe1' + (len(xmp) + 2).to_bytes(2, 'big') + xmp
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            f.write(data[:pos] + segment + data[pos:])
        os.replace(tmp, path)
        return True
    except Exception as e:
        _warn_once('jpeg', f'[Provenance] WARNING JPEG declaration not written: {e}')
        return False


def watermark_wanted():
    return str(_setting('watermark', False)).strip().lower() in ('1', 'true', 'on', 'yes')


def trustmark_available():
    return importlib.util.find_spec('trustmark') is not None


def watermark_id():
    ident = str(_setting('watermark_id', 'Fooocus26') or 'Fooocus26')
    ident = ident.encode('ascii', 'ignore').decode('ascii')[:WM_MAX_CHARS]
    return ident or 'Fooocus26'


def _tm():
    global _TM
    if _TM is None:
        from trustmark import TrustMark
        # CPU force : le GPU reste a la generation (~4 s d'init, ~0.1 s par image)
        _TM = TrustMark(verbose=False, model_type='Q', device='cpu', loadRemover=False)
    return _TM


def maybe_watermark(image):
    """Filigrane TrustMark si demande et disponible, sinon l'image intacte. La
    sauvegarde ne doit jamais echouer a cause de la provenance."""
    if not watermark_wanted():
        return image
    if not trustmark_available():
        _warn_once('tm-missing', '[Provenance] provenance.watermark is on but the trustmark '
                                 'package is missing (pip install trustmark): images '
                                 'saved without a watermark, with the XMP declaration.')
        return image
    try:
        alpha = image.getchannel('A') if image.mode == 'RGBA' else None
        out = _tm().encode(image.convert('RGB'), watermark_id())
        if alpha is not None:
            out.putalpha(alpha)
        return out
    except Exception as e:
        _warn_once('tm-error', f'[Provenance] WARNING watermark not applied: {e}')
        return image


# -------------------------------------------------------------------- lecture

def _xmp_text(image):
    info = getattr(image, 'info', None) or {}
    for key in (PNG_XMP_KEY, 'xmp'):
        v = info.get(key)
        if v:
            return v.decode('utf-8', 'replace') if isinstance(v, (bytes, bytearray)) else str(v)
    for marker, payload in getattr(image, 'applist', None) or []:
        if marker == 'APP1' and payload.startswith(JPEG_XMP_HEADER):
            return payload[len(JPEG_XMP_HEADER):].decode('utf-8', 'replace')
    return ''


def _read_c2pa(path):
    if not path or importlib.util.find_spec('c2pa') is None:
        return None
    try:
        import json
        import c2pa
        with c2pa.Reader(path) as r:
            data = json.loads(r.json())
            state = ''
            try:
                state = str(r.get_validation_state() or '')
            except Exception:
                pass
            active = data.get('manifests', {}).get(data.get('active_manifest', ''), {})
            sig = active.get('signature_info') or {}
            return {'generator': active.get('claim_generator', ''), 'issuer': sig.get('issuer', ''),
                    'when': sig.get('time', ''), 'state': state}
    except Exception:
        return None


def describe(image_or_path, check_watermark=None):
    """Section 'provenance' pour l'onglet Metadata (dict JSON). check_watermark None =
    decoder TrustMark seulement s'il est installe."""
    path = image_or_path if isinstance(image_or_path, str) else getattr(image_or_path, 'filename', '') or ''
    image = image_or_path
    opened = None
    try:
        if isinstance(image_or_path, str):
            from PIL import Image
            opened = image = Image.open(image_or_path)
        out = {}
        xmp = _xmp_text(image)
        m = re.search(r'DigitalSourceType(?:="|>)([^"<]+)', xmp)
        if m and m.group(1).rstrip('/').endswith('trainedAlgorithmicMedia'):
            out['ai_generated'] = 'declared (IPTC DigitalSourceType: trainedAlgorithmicMedia)'
        elif m:
            out['digital_source_type'] = m.group(1)
        else:
            out['ai_generated'] = 'no machine-readable declaration found'
        tool = re.search(r'CreatorTool(?:="|>)([^"<]+)', xmp)
        if tool:
            out['generator'] = tool.group(1)
        c2 = _read_c2pa(path if os.path.isfile(path) else None)
        if c2:
            out['c2pa'] = c2
        elif importlib.util.find_spec('c2pa') is None:
            out['c2pa'] = 'not checked (pip install c2pa-python)'
        want_wm = trustmark_available() if check_watermark is None else check_watermark
        if want_wm and trustmark_available():
            try:
                secret, present, _schema = _tm().decode(image.convert('RGB'))
                out['watermark'] = f'TrustMark detected: {secret}' if present else 'no TrustMark watermark'
            except Exception as e:
                out['watermark'] = f'not checked ({e})'
        elif check_watermark:
            out['watermark'] = 'not checked (pip install trustmark)'
        out['note'] = NOT_A_PROOF
        return out
    finally:
        if opened is not None:
            opened.close()
