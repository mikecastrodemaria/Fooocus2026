"""custom-33 — Exact face swap on every output, against a stub Face swap plugin (stdlib + numpy/PIL).

Run:  py -3.10 -m unittest tests.test_exact_faceswap -v
"""
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

from modules import exact_faceswap as X
from modules import global_faceswap as G

# A plugin whose "Face swap" action copies the input to the output folder after painting
# the top-left pixel white, so a swap is observable. Exit code 1 when asked to fail.
STUB = r'''
import os, sys
from PIL import Image
args = sys.argv[1:]
def val(flag):
    return args[args.index(flag) + 1]
if os.environ.get('STUB_FAIL'):
    print('no face detected in the input', file=sys.stderr); sys.exit(1)
assert '--faceswap-only' in args, args
assert os.path.isfile(val('--faceswap-src')), 'source face missing'
im = Image.open(val('-i')).convert('RGB')
im.putpixel((0, 0), (255, 255, 255))
out = os.path.join(val('--output-dir'), 'swapped.png')
im.save(out)
print(os.path.abspath(out))
'''


def manifest(with_faceswap=True):
    m = {
        'manifest_version': 1, 'id': 'studio', 'name': 'Studio stub', 'version': '1',
        'entry': {
            'command': [sys.executable, 'stub.py', '--cli'], 'input_arg': '-i',
            'output': {'mode': 'print_output', 'flag': '--print-output',
                       'save_mode_flag': ['--save-mode', 'custom'],
                       'save_dir_flag': ['--output-dir', '{output_dir}'],
                       'format_flag': ['--output-format', 'png']},
        },
        'params': [{'key': 'factor', 'type': 'slider', 'arg': '--factor'}],
        'env': {'strategies': {'fresh_venv': {'create': [], 'steps': []}}},
    }
    if with_faceswap:
        m['actions'] = [
            {'id': 'upscale', 'label': 'Upscale'},
            {'id': 'faceswap', 'label': 'Face swap', 'args': ['--faceswap-only'], 'params': [],
             'image_params': [{'key': 'faceswap_src', 'label': 'Source face', 'arg': '--faceswap-src'}]},
        ]
    return m


class World(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='exact-')
        self.plugins = os.path.join(self.root, 'installed')
        os.makedirs(self.plugins)
        self.cfg = types.ModuleType('modules.config')
        self.cfg.config_path = os.path.join(self.root, 'config.txt')
        self.cfg.global_faceswap_config = {}
        self.cfg.exact_faceswap_config = {}
        self._old = sys.modules.get('modules.config')
        sys.modules['modules.config'] = self.cfg
        X.INSTALL_ROOT_OVERRIDE = self.plugins
        X._warned['msg'] = None
        self.logs = []
        os.environ.pop('STUB_FAIL', None)

    def tearDown(self):
        X.INSTALL_ROOT_OVERRIDE = None
        os.environ.pop('STUB_FAIL', None)
        if self._old is None:
            sys.modules.pop('modules.config', None)
        else:
            sys.modules['modules.config'] = self._old
        shutil.rmtree(self.root, ignore_errors=True)

    def install(self, with_faceswap=True):
        d = os.path.join(self.plugins, 'studio')
        os.makedirs(d)
        with open(os.path.join(d, 'fooocus_extra.json'), 'w', encoding='utf-8') as f:
            json.dump(manifest(with_faceswap), f)
        with open(os.path.join(d, 'stub.py'), 'w', encoding='utf-8') as f:
            f.write(STUB)
        return d

    def face(self):
        G.save_face(np.full((16, 16, 3), 90, np.uint8))

    def image(self):
        return np.full((12, 10, 3), 7, np.uint8)


class TestSettings(World):
    def test_defaults(self):
        self.assertEqual(X.settings(), {'enabled': False, 'keep_original': False,
                                        'offload_host': False, 'timeout': 600})

    def test_save_writes_block_and_keeps_other_keys(self):
        with open(self.cfg.config_path, 'w', encoding='utf-8') as f:
            json.dump({'global_faceswap': {'enabled': True}}, f)
        X.save(True, True, False)
        with open(self.cfg.config_path, encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data['global_faceswap'], {'enabled': True})
        self.assertEqual(data['exact_faceswap'], {'enabled': True, 'keep_original': True,
                                                  'offload_host': False, 'timeout': 600})
        self.assertEqual(self.cfg.exact_faceswap_config, data['exact_faceswap'])

    def test_timeout_clamped(self):
        self.cfg.exact_faceswap_config = {'timeout': 1}
        self.assertEqual(X.settings()['timeout'], 30)


class TestAvailability(World):
    def test_no_plugin(self):
        ok, msg = X.availability()
        self.assertFalse(ok)
        self.assertIn('crispz-studio', msg)

    def test_plugin_without_faceswap_action(self):
        self.install(with_faceswap=False)
        self.assertIsNone(X.find_action())
        ok, msg = X.availability()
        self.assertFalse(ok)
        self.assertIn('older version', msg, 'a studio without the action is an outdated install')
        self.assertIn('Updates', msg)

    def test_other_plugin_only_names_it(self):
        d = self.install(with_faceswap=False)
        m = manifest(with_faceswap=False)
        m['id'] = 'crispz'
        with open(os.path.join(d, 'fooocus_extra.json'), 'w', encoding='utf-8') as f:
            json.dump(m, f)
        ok, msg = X.availability()
        self.assertFalse(ok)
        self.assertIn('crispz', msg)
        self.assertIn('install crispz-studio', msg)

    def test_plugin_but_no_face(self):
        self.install()
        ok, msg = X.availability()
        self.assertFalse(ok)
        self.assertIn('no face', msg)

    def test_ready(self):
        self.install()
        self.face()
        ok, msg = X.availability()
        self.assertTrue(ok, msg)
        self.assertIn('Face swap', msg)

    def test_status_html_states(self):
        self.assertIn('off', X.status_html())
        X.save(True, False, False)
        self.assertIn('not possible', X.status_html())
        self.install()
        self.face()
        self.assertIn('active', X.status_html())


class TestSwap(World):
    def test_off_returns_the_image_untouched(self):
        img = self.image()
        out, swapped = X.maybe_swap(img, log=self.logs.append)
        self.assertIs(out, img)
        self.assertFalse(swapped)
        self.assertEqual(self.logs, [])

    def test_enabled_but_unavailable_warns_once(self):
        X.save(True, False, False)
        img = self.image()
        for _ in range(3):
            out, swapped = X.maybe_swap(img, log=self.logs.append)
            self.assertIs(out, img)
            self.assertFalse(swapped)
        self.assertEqual(len(self.logs), 1)
        self.assertIn('skipped', self.logs[0])

    def test_swap_runs_the_plugin_with_the_face(self):
        self.install()
        self.face()
        X.save(True, False, False)
        out, swapped = X.maybe_swap(self.image(), log=self.logs.append)
        self.assertTrue(swapped)
        self.assertEqual(out.shape, (12, 10, 3))
        self.assertEqual(tuple(out[0, 0]), (255, 255, 255), 'the stub painted its marker')
        self.assertEqual(tuple(out[5, 5]), (7, 7, 7), 'the rest of the image is kept')
        self.assertTrue(any('face pasted' in ln for ln in self.logs))

    def test_plugin_failure_keeps_the_image(self):
        self.install()
        self.face()
        X.save(True, False, False)
        os.environ['STUB_FAIL'] = '1'
        img = self.image()
        out, swapped = X.maybe_swap(img, log=self.logs.append)
        self.assertIs(out, img)
        self.assertFalse(swapped)
        self.assertTrue(any('no face detected' in ln and 'kept as is' in ln for ln in self.logs))

    def test_temp_files_are_cleaned(self):
        self.install()
        self.face()
        X.save(True, False, False)
        before = set(os.listdir(tempfile.gettempdir()))
        X.maybe_swap(self.image(), log=self.logs.append)
        after = set(os.listdir(tempfile.gettempdir()))
        self.assertFalse([d for d in after - before if d.startswith('exact-faceswap-')])


if __name__ == '__main__':
    unittest.main()
