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
            out = C.search_models('ink', types='LORA', base_model='SDXL 1.0', api_key='K', all_architectures=True)
        params = api.call_args[0][1]
        self.assertNotIn('baseModels', params, 'no base filter when every architecture is asked for')
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

    def test_only_fooocus_architectures_by_default(self):
        with mock.patch.object(C, '_api_request', return_value=API_REPLY) as api:
            out = C.search_models('ink', base_model='SDXL 1.0')
        self.assertEqual(api.call_args[0][1]['baseModels'], C.COMPATIBLE_BASES)
        self.assertNotIn('Flux.1 D', C.COMPATIBLE_BASES)
        self.assertEqual([c['versionId'] for c in out], [102, 201], 'the Flux version is dropped even if the API returns it')

    def test_base_support_levels(self):
        self.assertEqual(C.base_model_support('Pony V7')[0], 'hidden')
        self.assertEqual(C.base_model_support('Pony V6')[0], 'ok')
        self.assertEqual(C.base_model_support('Pony')[0], 'ok')
        self.assertEqual(C.base_model_support('Illustrious')[0], 'ok')
        self.assertEqual(C.base_model_support('SDXL Lightning')[0], 'ok')
        self.assertEqual(C.base_model_support('SD 1.5')[0], 'sd15')
        self.assertEqual(C.base_model_support('SD 3.5')[0], 'hidden')
        self.assertEqual(C.base_model_support('')[0], 'unknown')


class TestFolders(unittest.TestCase):
    """Rangement <racine>/<base>/<categorie> dans les dossiers qui existent deja."""
    TREE = ['_SDXL_1_0/style', '_SDXL_1_0/actor', '_SDXL_1_0/.nsfw', '_SDXL_1_0/.nsfw_style',
            '_SDXL_1_0/helper', '_SDXL_1_0/accelerator', '_Pony/style', '_Pony/.nsfw',
            '_Illustrous/style', '_SD_1.5/style', '_Flux/_Flux_style', '_trash/old', 'Other/tool']

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='fciv_tree_')
        for d in self.TREE:
            os.makedirs(os.path.join(self.root, *d.split('/')))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def sub(self, base, tags=(), nsfw=False):
        return C.suggest_subfolder({'baseModel': base, 'tags': list(tags), 'nsfw': nsfw}, self.root)[0]

    def test_base_then_category_from_tags(self):
        self.assertEqual(self.sub('SDXL 1.0', ['ink', 'style']), '_SDXL_1_0/style')
        self.assertEqual(self.sub('SDXL 1.0', ['character', 'woman']), '_SDXL_1_0/actor')
        self.assertEqual(self.sub('SDXL Lightning', ['lightning']), '_SDXL_1_0/accelerator')
        self.assertEqual(self.sub('Pony', ['style']), '_Pony/style')
        self.assertEqual(self.sub('SDXL 1.0', ['landscape']), '_SDXL_1_0', 'no category tag: base folder only')

    def test_nsfw_goes_to_the_nsfw_folders(self):
        self.assertEqual(self.sub('SDXL 1.0', ['style'], nsfw=True), '_SDXL_1_0/.nsfw_style')
        self.assertEqual(self.sub('SDXL 1.0', ['character'], nsfw=True), '_SDXL_1_0/.nsfw')
        self.assertEqual(self.sub('SDXL 1.0', ['style']), '_SDXL_1_0/style', 'SFW never lands in .nsfw_style')

    def test_folder_names_are_matched_loosely_but_never_sdxl_for_sd15(self):
        self.assertEqual(self.sub('Illustrious', ['style']), '_Illustrous/style')
        self.assertEqual(self.sub('SD 1.5', ['style']), '_SD_1.5/style')
        self.assertEqual(self.sub('Flux.1 D', ['style']), '_Flux/_Flux_style')

    def test_unknown_base_falls_back_to_other_then_root(self):
        self.assertEqual(self.sub('SD 3.5', ['tool']), 'Other/tool')
        shutil.rmtree(os.path.join(self.root, 'Other'))
        self.assertEqual(C.suggest_subfolder({'baseModel': 'SD 3.5'}, self.root),
                         ('', 'no matching folder, models root'))

    def test_list_and_resolve_subfolders(self):
        subs = C.list_subfolders(self.root)
        self.assertIn('_SDXL_1_0/.nsfw', subs)
        self.assertFalse(any(s.startswith('_trash') for s in subs))
        self.assertEqual(C.resolve_subfolder(self.root, '(root)'), self.root)
        self.assertEqual(C.resolve_subfolder(self.root, '_Pony\\style'), os.path.join(self.root, '_Pony', 'style'))
        self.assertEqual(C.resolve_subfolder(self.root, '/new/'), os.path.join(self.root, 'new'),
                         'a leading slash stays inside the models folder')
        for bad in ('../escape', 'style/../../escape', 'C:/Windows'):
            with self.assertRaises(ValueError):
                C.resolve_subfolder(self.root, bad)


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

    def test_a_file_already_filed_in_another_subfolder_is_not_downloaded_again(self):
        os.makedirs(os.path.join(self.dir, '_SDXL_1_0', 'style'))
        mine = os.path.join(self.dir, '_SDXL_1_0', 'style', 'Demo.safetensors')
        with open(mine, 'wb') as f:
            f.write(b'mine')
        seen = len(_Handler.seen)
        res = C.download_model_file(self.cand(), os.path.join(self.dir, '_SDXL_1_0'), search_root=self.dir)
        self.assertTrue(res['success'])
        self.assertIn('already exists in', res['message'])
        self.assertEqual((res['path'], len(_Handler.seen)), (mine, seen))

    def test_a_locked_file_without_key_says_so(self):
        res = C.download_model_file(self.cand(path='/locked'), self.dir)
        self.assertFalse(res['success'])
        self.assertIn('HTTP 403', res['message'])
        self.assertIn('API key', res['message'])
        self.assertEqual(os.listdir(self.dir), [])


if __name__ == '__main__':
    unittest.main()
