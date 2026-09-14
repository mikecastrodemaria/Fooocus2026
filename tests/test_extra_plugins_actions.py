"""custom-24 — plugins Extra : plusieurs actions par plugin (ex. Upscale + Face swap).

Run:  py -3.10 -m unittest tests.test_extra_plugins_actions -v
"""
import copy
import json
import os
import shutil
import tempfile
import unittest

from extra_plugins import manifest as M, runner

BASE = {
    'manifest_version': 1, 'id': 'crispz-studio', 'name': 'crispz-studio', 'version': '1.16.0',
    'entry': {'command': ['{venv_python}', 'app.py', '--cli'], 'input_arg': '-i',
              'output': {'mode': 'print_output', 'flag': '--print-output',
                         'save_mode_flag': ['--save-mode', 'custom'],
                         'save_dir_flag': ['--output-dir', '{output_dir}']},
              'models_arg': ['--esrgan-dir', '{esrgan_dir}']},
    'params': [
        {'key': 'model', 'type': 'dropdown', 'arg': '-m', 'choices_cmd': ['--list-models']},
        {'key': 'factor', 'type': 'slider', 'arg': '--factor', 'default': 2.0},
    ],
    'server': {'launch': ['{venv_python}', 'app.py', '--serve', '--port', '7861']},
}


def with_actions(actions):
    m = copy.deepcopy(BASE)
    m['actions'] = actions
    return m


FACESWAP = {'id': 'faceswap', 'label': 'Face swap', 'args': ['--faceswap-only'], 'params': [],
            'image_params': [{'key': 'faceswap_src', 'label': 'Source face', 'arg': '--faceswap-src'}]}


class TestActions(unittest.TestCase):
    def test_a_manifest_without_actions_keeps_the_single_upscale(self):
        acts = M.actions(BASE)
        self.assertEqual(len(acts), 1)
        self.assertEqual((acts[0]['id'], acts[0]['params'], acts[0]['server']),
                         ('upscale', ['model', 'factor'], True))

    def test_declared_actions_are_normalised(self):
        m = with_actions([{'id': 'upscale', 'label': 'Upscale', 'server': True}, FACESWAP])
        M.validate(m)
        up, fs = M.actions(m)
        self.assertEqual(up['params'], ['model', 'factor'], 'params absents = tous les params')
        self.assertTrue(up['server'])
        self.assertEqual((fs['params'], fs['args'], fs['server']), ([], ['--faceswap-only'], False),
                         'une action a flags ne passe pas par /upscale')
        self.assertEqual(fs['image_params'][0]['arg'], '--faceswap-src')

    def test_invalid_actions_are_refused(self):
        bad = [
            [],
            [{'label': 'no id'}],
            [{'id': 'a', 'label': 'A'}, {'id': 'a', 'label': 'again'}],
            [{'id': 'a', 'label': 'A', 'params': ['nope']}],
            [{'id': 'a', 'label': 'A', 'image_params': [{'key': 'x'}]}],
            [{'id': 'a', 'label': 'A', 'args': '--flag'}],
        ]
        for actions in bad:
            with self.assertRaises(M.ManifestError, msg=actions):
                M.validate(with_actions(actions))

    def test_the_command_carries_action_flags_and_extra_images(self):
        cmd = runner.build_upscale_command(
            with_actions([FACESWAP]), 'P', 'C:/in.png', 'C:/out', {}, esrgan_dir=None,
            extra_args=['--faceswap-only'], image_args=[('--faceswap-src', 'C:/face.png')])
        self.assertEqual(cmd[3:8], ['-i', 'C:/in.png', '--faceswap-only', '--faceswap-src', 'C:/face.png'])
        self.assertIn('--print-output', cmd)
        self.assertNotIn('--esrgan-dir', cmd)

    def test_the_crispz_studio_manifest_is_valid(self):
        path = os.path.join('..', 'crispz-studio', 'fooocus_extra.json')
        if not os.path.isfile(path):
            self.skipTest('crispz-studio non present a cote de Fooocus2026')
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        M.validate(data)
        self.assertEqual([a['id'] for a in M.actions(data)], ['upscale', 'faceswap'])


class TestPluginTabBuilds(unittest.TestCase):
    """L'onglet se construit (Gradio 3.41) avec une ou plusieurs actions."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='fact_')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def build(self, manifest):
        import gradio as gr
        from extra_plugins import ui
        plugin = {'id': manifest['id'], 'name': manifest['name'], 'version': manifest['version'],
                  'dir': self.dir, 'manifest': manifest}
        with gr.Blocks() as demo:
            ui._build_plugin_tab(plugin)
        return demo

    def test_single_and_multiple_actions(self):
        self.build(BASE)
        demo = self.build(with_actions([{'id': 'upscale', 'label': 'Upscale'}, FACESWAP]))
        labels = [getattr(b, 'label', None) for b in demo.blocks.values()]
        self.assertIn('Source face', labels)
        self.assertIn('Face swap', [getattr(b, 'value', None) for b in demo.blocks.values()])


if __name__ == '__main__':
    unittest.main()
