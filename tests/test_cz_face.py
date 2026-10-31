"""crispz-sdxl: cz_face.py on CPU. The fake detector (deterministic) and the
pure helpers; the insightface path is exercised with a stub analyser so the
mapping of its objects (bbox, kps, det_score, normed_embedding) is checked
without the model.

Run:  python -m pytest tests/test_cz_face.py -q
"""
import os
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.argv = sys.argv[:1]

import cz_face  # noqa: E402


class TestFakeDetector(unittest.TestCase):
    def test_one_centred_face_with_a_stable_normed_embedding(self):
        with mock.patch.dict(os.environ, {'CRISPZ_FAKE_ENGINE': '1'}):
            image = Image.new('RGB', (400, 200), (1, 2, 3))
            faces = cz_face.detect_faces_full(image)
            self.assertEqual(len(faces), 1)
            self.assertEqual(faces[0]['box'], [100, 50, 300, 150])
            self.assertEqual(faces[0]['mouth'], [200, 125])
            self.assertEqual(len(faces[0]['embedding']), cz_face.EMBEDDING_SIZE)
            self.assertAlmostEqual(np.linalg.norm(faces[0]['embedding']), 1.0, places=6)
            self.assertEqual(cz_face.detect_faces(np.array(image)), [[100, 50, 300, 150]])
            other = cz_face.detect_faces_full(Image.new('RGB', (400, 200), (3, 2, 1)))
            self.assertLess(cz_face.similarity(faces[0]['embedding'], other[0]['embedding']), 0.5)
            self.assertAlmostEqual(cz_face.similarity(faces[0]['embedding'], faces[0]['embedding']), 1.0, places=6)
            self.assertTrue(cz_face.available())


class _Face:
    def __init__(self, bbox, kps, score, emb):
        self.bbox = np.array(bbox, dtype=float)
        self.kps = np.array(kps, dtype=float)
        self.det_score = score
        self.normed_embedding = np.array(emb, dtype=float)


class TestInsightfaceMapping(unittest.TestCase):
    def test_faces_are_mapped_and_sorted_biggest_first(self):
        small = _Face([10, 10, 20, 20], [[12, 12], [18, 12], [15, 15], [13, 18], [17, 18]], 0.7, [1.0] + [0.0] * 511)
        big = _Face([100.4, 50.6, 300, 250], [[150, 100], [250, 100], [200, 150], [170, 200], [230, 200]], 0.9,
                    [0.0, 1.0] + [0.0] * 510)
        app = mock.Mock()
        app.get.return_value = [small, big]
        with mock.patch.dict(os.environ, {'CRISPZ_FAKE_ENGINE': '0'}), \
                mock.patch.object(cz_face, '_analyser', app):
            faces = cz_face.detect_faces_full(np.zeros((300, 400, 3), dtype=np.uint8))
        self.assertEqual([f['box'] for f in faces], [[100, 51, 300, 250], [10, 10, 20, 20]])
        self.assertEqual(faces[0]['mouth'], [200, 200])
        self.assertEqual((faces[0]['score'], faces[0]['embedding'][1]), (0.9, 1.0))
        bgr = app.get.call_args[0][0]
        self.assertEqual(bgr.shape, (300, 400, 3))

    def test_without_insightface_the_error_says_how_to_proceed(self):
        with mock.patch.dict(os.environ, {'CRISPZ_FAKE_ENGINE': '0'}), \
                mock.patch.object(cz_face, '_analyser', None), \
                mock.patch.dict(sys.modules, {'insightface': None, 'insightface.app': None}):
            with self.assertRaises(RuntimeError) as cm:
                cz_face.detect_faces_full(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertIn('CRISPZ_FAKE_ENGINE=1', str(cm.exception))


if __name__ == '__main__':
    unittest.main()
