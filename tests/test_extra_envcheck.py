"""custom-30 — Detection of an interrupted torch install in a plugin venv (stdlib only).

Run:  py -3.10 -m unittest tests.test_extra_envcheck -v
"""
import os
import shutil
import tempfile
import unittest

from extra_plugins import envcheck as E

MANIFEST = {
    'id': 'demo',
    'env': {'strategies': {
        'fresh_venv': {'create': [], 'steps': [
            {'name': 'pip', 'cmd': ['{venv_python}', '-m', 'pip', 'install', '-U', 'pip']},
            {'name': 'torch cu128', 'cmd': ['{venv_pip}', 'install', 'torch==2.7.1', 'torchvision',
                                            '--index-url', 'https://download.pytorch.org/whl/cu128']},
            {'name': 'deps', 'cmd': ['{venv_pip}', 'install', '-r', 'requirements.txt']},
        ]},
        'reuse_python': {'create': [], 'steps': [
            {'name': 'deps', 'cmd': ['{venv_pip}', 'install', '-r', 'requirements.txt']},
        ]},
    }},
}
NO_TORCH_MANIFEST = {'id': 'light', 'env': {'strategies': {'fresh_venv': {'steps': [
    {'name': 'deps', 'cmd': ['{venv_pip}', 'install', 'pillow']}]}}}}

WINERROR = (
    'Traceback (most recent call last):\n'
    '  File "app.py", line 28, in <module>\n    import torch\n'
    'OSError: [WinError 127] La procédure spécifiée est introuvable. Error loading '
    '"C:\\x\\crispz\\.venv\\lib\\site-packages\\torch\\lib\\cudnn_cnn64_9.dll" or one of its dependencies.')


class VenvWorld(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='envcheck-')
        self.plugin = os.path.join(self.root, 'crispz')
        os.makedirs(self.plugin)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def venv(self, layout='win', system_site=False):
        venv = os.path.join(self.plugin, '.venv')
        sp = (os.path.join(venv, 'Lib', 'site-packages') if layout == 'win'
              else os.path.join(venv, 'lib', 'python3.10', 'site-packages'))
        os.makedirs(sp)
        with open(os.path.join(venv, 'pyvenv.cfg'), 'w') as f:
            f.write('home = /usr/bin\ninclude-system-site-packages = %s\nversion = 3.10.20\n'
                    % ('true' if system_site else 'false'))
        return sp

    def torch(self, sp, dist_info=True, record=True):
        os.makedirs(os.path.join(sp, 'torch', 'lib'))
        if dist_info:
            d = os.path.join(sp, 'torch-2.7.1+cu128.dist-info')
            os.makedirs(d)
            if record:
                open(os.path.join(d, 'RECORD'), 'w').close()


class TestTorchState(VenvWorld):
    def test_no_venv(self):
        self.assertEqual(E.torch_state(self.plugin), 'no-venv')

    def test_missing(self):
        self.venv()
        self.assertEqual(E.torch_state(self.plugin), 'missing')

    def test_incomplete_folder_without_dist_info(self):
        sp = self.venv()
        self.torch(sp, dist_info=False)
        self.assertEqual(E.torch_state(self.plugin), 'incomplete')

    def test_incomplete_dist_info_without_record(self):
        sp = self.venv()
        self.torch(sp, record=False)
        self.assertEqual(E.torch_state(self.plugin), 'incomplete')

    def test_ok(self):
        sp = self.venv()
        self.torch(sp)
        self.assertEqual(E.torch_state(self.plugin), 'ok')

    def test_ok_posix_layout(self):
        sp = self.venv(layout='posix')
        self.torch(sp)
        self.assertEqual(E.torch_state(self.plugin), 'ok')

    def test_inherited_with_system_site_packages(self):
        self.venv(system_site=True)
        self.assertEqual(E.torch_state(self.plugin), 'inherited')


class TestPreflight(VenvWorld):
    def test_plugin_without_torch_step_is_never_checked(self):
        self.assertIsNone(E.preflight(self.plugin, NO_TORCH_MANIFEST))

    def test_ok_venv_passes(self):
        self.torch(self.venv())
        self.assertIsNone(E.preflight(self.plugin, MANIFEST))

    def test_inherited_passes(self):
        self.venv(system_site=True)
        self.assertIsNone(E.preflight(self.plugin, MANIFEST))

    def test_incomplete_names_the_cause_and_the_manifest_command(self):
        self.torch(self.venv(), dist_info=False)
        msg = E.preflight(self.plugin, MANIFEST)
        self.assertIn('torch is incomplete', msg)
        self.assertIn('dist-info', msg)
        self.assertIn('install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128', msg)
        self.assertIn(self.plugin, msg)
        self.assertNotIn('{venv_pip}', msg, 'placeholders are rendered')
        self.assertIn('Install with Force', msg)

    def test_missing_torch_is_reported(self):
        self.venv()
        msg = E.preflight(self.plugin, MANIFEST)
        self.assertIn('torch is not installed', msg)
        self.assertIn('install torch==2.7.1', msg)

    def test_no_venv_points_to_the_manager(self):
        msg = E.preflight(self.plugin, MANIFEST)
        self.assertIn('.venv missing', msg)
        self.assertIn('Manager', msg)


class TestManifest(unittest.TestCase):
    def test_torch_step_found_by_token(self):
        name, step = E.torch_step(MANIFEST)
        self.assertEqual(name, 'fresh_venv')
        self.assertEqual(step['name'], 'torch cu128')

    def test_torch_token_shapes(self):
        for tok in ('torch', 'torch==2.7.1', 'torch>=2', 'torch<3', 'torch~=2.7'):
            self.assertTrue(E.needs_torch({'env': {'strategies': {'s': {'steps': [{'cmd': ['pip', 'install', tok]}]}}}}), tok)
        for tok in ('torchvision', 'pytorch-lightning', 'torchaudio==2'):
            self.assertFalse(E.needs_torch({'env': {'strategies': {'s': {'steps': [{'cmd': ['pip', 'install', tok]}]}}}}), tok)

    def test_empty_or_odd_manifests(self):
        self.assertFalse(E.needs_torch(None))
        self.assertFalse(E.needs_torch({}))
        self.assertFalse(E.needs_torch({'env': {'strategies': {'s': None}}}))

    def test_repair_command_quotes_paths_with_spaces(self):
        cmd = E.repair_command(MANIFEST, os.path.join('C:' + os.sep, 'my plugins', 'crispz'))
        self.assertIn('"', cmd)
        self.assertIn('install torch==2.7.1', cmd)

    def test_repair_command_without_step_is_generic(self):
        cmd = E.repair_command(NO_TORCH_MANIFEST, '/p')
        self.assertTrue(cmd.endswith('install torch'))


class TestExplainFailure(unittest.TestCase):
    def test_winerror_127_on_a_torch_dll(self):
        hint = E.explain_failure(WINERROR, MANIFEST, '/p')
        self.assertIn('torch DLL failed to load', hint)
        self.assertIn('interrupted torch install', hint)
        self.assertIn('install torch==2.7.1', hint)

    def test_missing_module(self):
        self.assertIn('not importable', E.explain_failure("ModuleNotFoundError: No module named 'torch'"))

    def test_unrelated_error_gives_nothing(self):
        self.assertEqual(E.explain_failure('FileNotFoundError: model.pth'), '')
        self.assertEqual(E.explain_failure(''), '')
        self.assertEqual(E.explain_failure(None), '')


if __name__ == '__main__':
    unittest.main()
