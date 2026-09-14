"""custom-22 — recherche et telechargement CivitAI, sans reseau.

La recherche lit une reponse /models factice ; le telechargement parle a un vrai petit
serveur HTTP local (stdlib) : verification SHA256, fichier corrompu supprime, rien
d'ecrase, cle API requise annoncee.

Run:  py -3.10 -m unittest tests.test_civitai_search -v
"""
import hashlib
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

# civitai_api -> hash_cache -> args_manager analyse sys.argv a l'import : sans ca il
# lirait les arguments d'unittest ("tests.test_civitai_search -v") et quitterait.
sys.argv = sys.argv[:1]

from modules import civitai_api as C  # noqa: E402

PAYLOAD = b'safetensors-bytes-' * 20000  # ~360 KB
SHA = hashlib.sha256(PAYLOAD).hexdigest()

API_REPLY = {'items': [
    {'id': 11, 'name': 'Ink Style', 'type': 'LORA', 'nsfw': False, 'creator': {'username': 'mike'},
     'modelVersions': [
         {'id': 101, 'name': 'Flux v1', 'baseModel': 'Flux.1 D', 'trainedWords': ['ink'],
          'files': [{'name': 'ink_flux.safetensors', 'sizeKB': 1024 * 18, 'primary': True,
                     'downloadUrl': 'https://civitai.com/api/download/models/101',
                     'hashes': {'SHA256': 'AB' * 32}}],
          'images': [{'url': 'https://img/nsfw.jpg', 'nsfwLevel': 8},
                     {'url': 'https://img/ok.jpg', 'nsfwLevel': 1}]},
         {'id': 102, 'name': 'XL v2', 'baseModel': 'SDXL 1.0', 'trainedWords': [],
          'files': [{'name': 'ink_xl.safetensors', 'sizeKB': 200, 'primary': True}],
          'images': []},
     ]},
    {'id': 12, 'name': 'Old', 'type': 'LORA', 'modelVersions': [
        {'id': 201, 'name': 'v1', 'baseModel': 'SD 1.5', 'files': []}]},
]}


class TestSearch(unittest.TestCase):
    def test_versions_are_flattened_and_the_wanted_base_comes_first(self):
        with mock.patch.object(C, '_api_request', return_value=API_REPLY) as api:
            out = C.search_models('ink', types='LORA', base_model='SDXL 1.0', api_key='K')
        params = api.call_args[0][1]
        self.assertEqual((params['query'], params['types'], params['nsfw']), ('ink', 'LORA', 'false'))
        self.assertEqual(api.call_args[1]['api_key'], 'K')
        self.assertEqual([c['versionId'] for c in out], [102, 101, 201])
        flux = out[1]
        self.assertEqual((flux['support'], flux['sha256'], flux['trainedWords']), ('hidden', 'ab' * 32, ['ink']))
        self.assertEqual(flux['previewUrl'], 'https://img/ok.jpg', 'pas de preview NSFW sans la case')
        self.assertEqual(out[2]['support'], 'sd15')
        self.assertTrue(C.candidate_label(1, flux).startswith('2. ⛔ Ink Style — Flux v1 [Flux.1 D] 18 MB'))

    def test_empty_query_or_network_failure_returns_nothing(self):
        self.assertEqual(C.search_models('  '), [])
        with mock.patch.object(C, '_api_request', return_value=None):
            self.assertEqual(C.search_models('ink'), [])

    def test_base_support_levels(self):
        self.assertEqual(C.base_model_support('Pony')[0], 'ok')
        self.assertEqual(C.base_model_support('Illustrious')[0], 'ok')
        self.assertEqual(C.base_model_support('SDXL Lightning')[0], 'ok')
        self.assertEqual(C.base_model_support('SD 1.5')[0], 'sd15')
        self.assertEqual(C.base_model_support('SD 3.5')[0], 'hidden')
        self.assertEqual(C.base_model_support('')[0], 'unknown')


class _Handler(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        _Handler.seen.append(self.path)
        if self.path.startswith('/locked'):
            self.send_response(403, 'Forbidden')
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Length', str(len(PAYLOAD)))
        self.end_headers()
        self.wfile.write(PAYLOAD)


class TestDownload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
        cls.base = f'http://127.0.0.1:{cls.httpd.server_address[1]}'
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='fciv_')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def cand(self, path='/file', sha=SHA, name='demo.safetensors'):
        return {'downloadUrl': self.base + path, 'fileName': name, 'sha256': sha, 'sizeKB': len(PAYLOAD) / 1024}

    def test_a_verified_download_lands_with_its_hash_cached(self):
        steps = []
        res = C.download_model_file(self.cand(), self.dir, api_key='TOKEN', progress=lambda f, t: steps.append(f))
        self.assertTrue(res['success'], res)
        self.assertIn('verified', res['message'])
        with open(res['path'], 'rb') as f:
            self.assertEqual(f.read(), PAYLOAD)
        self.assertEqual(C._full_hash_cache[res['path']], SHA)
        self.assertEqual(steps[-1], 1.0)
        self.assertIn('token=TOKEN', _Handler.seen[-1])
        self.assertFalse(any(n.endswith('.part') for n in os.listdir(self.dir)))

    def test_a_corrupted_download_is_removed(self):
        res = C.download_model_file(self.cand(sha='00' * 32), self.dir)
        self.assertFalse(res['success'])
        self.assertIn('SHA256 mismatch', res['message'])
        self.assertEqual(os.listdir(self.dir), [])

    def test_an_existing_file_is_never_overwritten(self):
        dest = os.path.join(self.dir, 'demo.safetensors')
        with open(dest, 'wb') as f:
            f.write(b'mine')
        res = C.download_model_file(self.cand(), self.dir)
        self.assertTrue(res['success'])
        self.assertIn('not overwritten', res['message'])
        with open(dest, 'rb') as f:
            self.assertEqual(f.read(), b'mine')

    def test_a_locked_file_without_key_says_so(self):
        res = C.download_model_file(self.cand(path='/locked'), self.dir)
        self.assertFalse(res['success'])
        self.assertIn('HTTP 403', res['message'])
        self.assertIn('API key', res['message'])
        self.assertEqual(os.listdir(self.dir), [])


if __name__ == '__main__':
    unittest.main()
