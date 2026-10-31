"""crispz-sdxl: face detection and ArcFace embeddings, as in the family cz_face.py.

detect_faces(image) gives the boxes, detect_faces_full(image) the boxes, the
mouth point and the 512-float normed embedding of every face, biggest first.
The engine is insightface (buffalo_l: SCRFD detector + ArcFace), loaded once,
on the CPU provider unless onnxruntime has CUDA. Fooocus2026 does not ship
insightface (its face tools use facexlib under extras/), so the import is lazy
and a missing package gives a clear error, never a traceback at boot.

With the fake engine on (cz_core.fake_engine_enabled) a deterministic detector
answers instead: one face in the middle of the image, an embedding derived
from the pixels. It exists so cli_faces and its tests run on CPU without the
model; every call says so in the console.
"""
import hashlib
import importlib.util
import math

import numpy as np
from PIL import Image

import cz_core

MODEL = 'buffalo_l'
DET_SIZE = (640, 640)
EMBEDDING_SIZE = 512

_analyser = None
_fake_announced = False


def available():
    """True when a detector can answer: the fake one, or insightface installed."""
    return cz_core.fake_engine_enabled() or importlib.util.find_spec('insightface') is not None


def _to_rgb_array(image):
    if isinstance(image, str):
        with Image.open(image) as im:
            return np.array(im.convert('RGB'))
    if isinstance(image, Image.Image):
        return np.array(image.convert('RGB'))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[:, :, :3]


def _analyser_or_raise():
    global _analyser
    if _analyser is not None:
        return _analyser
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise RuntimeError(f'insightface is not installed ({e}): pip install insightface, or set '
                           'CRISPZ_FAKE_ENGINE=1 for a CPU dry run') from e
    providers = ['CPUExecutionProvider']
    try:
        import onnxruntime
        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    except ImportError:
        pass
    cz_core.log(f'loading insightface {MODEL} ({providers[0]})')
    app = FaceAnalysis(name=MODEL, providers=providers)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    _analyser = app
    return app


def _fake_faces(rgb):
    """One centred face, box = the middle half of the image, embedding from a
    hash of the pixels so the same image always gives the same vector."""
    global _fake_announced
    if not _fake_announced:
        cz_core.log('FAKE face detector active: one centred face per image, hashed embedding')
        _fake_announced = True
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = w // 4, h // 4, w - w // 4, h - h // 4
    digest = hashlib.sha256(rgb.tobytes()).digest()
    seed = int.from_bytes(digest[:8], 'big')
    vec = np.random.default_rng(seed).standard_normal(EMBEDDING_SIZE)
    vec = vec / np.linalg.norm(vec)
    return [{'box': [x1, y1, x2, y2], 'mouth': [(x1 + x2) // 2, y1 + int((y2 - y1) * 0.75)],
             'score': 0.99, 'embedding': [float(v) for v in vec], 'fake': True}]


def detect_faces_full(image):
    """[{box: [x1, y1, x2, y2], mouth: [x, y], score, embedding: [512 floats]}],
    biggest face first. image: path, PIL image or HWC uint8 array (RGB)."""
    rgb = _to_rgb_array(image)
    if cz_core.fake_engine_enabled():
        return _fake_faces(rgb)
    app = _analyser_or_raise()
    faces = app.get(np.ascontiguousarray(rgb[:, :, ::-1]))  # insightface wants BGR
    out = []
    for f in faces:
        x1, y1, x2, y2 = [int(round(float(v))) for v in f.bbox]
        kps = getattr(f, 'kps', None)
        if kps is not None and len(kps) >= 5:
            mouth = [int(round((kps[3][0] + kps[4][0]) / 2)), int(round((kps[3][1] + kps[4][1]) / 2))]
        else:
            mouth = [(x1 + x2) // 2, y1 + int((y2 - y1) * 0.75)]
        emb = getattr(f, 'normed_embedding', None)
        if emb is None:
            emb = f.embedding / max(np.linalg.norm(f.embedding), 1e-8)
        out.append({'box': [x1, y1, x2, y2], 'mouth': mouth, 'score': float(f.det_score),
                    'embedding': [float(v) for v in emb]})
    out.sort(key=lambda d: (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1]), reverse=True)
    return out


def detect_faces(image):
    """The boxes only, biggest first."""
    return [f['box'] for f in detect_faces_full(image)]


def similarity(a, b):
    """Cosine similarity of two embeddings (normed or not)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d and not math.isnan(d) else 0.0
