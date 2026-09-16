"""custom-32 — Global FaceSwap: settings, face file and injection into a task (numpy + PIL).

Run:  py -3.10 -m unittest tests.test_global_faceswap -v
"""
import json
import os
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np

from modules import global_faceswap as G

IP, CANNY, CPDS, FACE = 'ImagePrompt', 'PyraCanny', 'CPDS', 'FaceSwap'


class FakeTask:
    def __init__(self, tab='uov', input_image=True, mixing=(False, False), others=()):
        self.current_tab = tab
        self.input_image_checkbox = input_image
        self.mixing_image_prompt_and_vary_upscale, self.mixing_image_prompt_and_inpaint = mixing
        self.cn_tasks = {k: [] for k in (IP, CANNY, CPDS, FACE)}
        for k in others:
            self.cn_tasks[k].append([np.zeros((8, 8, 3), np.uint8), 0.5, 1.0])


class World(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='gfs-')
        self.cfg = types.ModuleType('modules.config')
        self.cfg.config_path = os.path.join(self.root, 'config.txt')
        self.cfg.global_faceswap_config = {}
        self._old = sys.modules.get('modules.config')
        sys.modules['modules.config'] = self.cfg
        self.logs = []

    def tearDown(self):
        if self._old is None:
            sys.modules.pop('modules.config', None)
        else:
            sys.modules['modules.config'] = self._old
        shutil.rmtree(self.root, ignore_errors=True)

    def face(self):
        rgb = np.zeros((32, 24, 3), np.uint8)
        rgb[..., 0] = 200
        return rgb

    def enable(self, stop=0.9, weight=0.75, with_face=True):
        return G.save(True, self.face() if with_face else None, stop, weight)


class TestSettingsAndFile(World):
    def test_defaults_when_nothing_configured(self):
        self.assertEqual(G.settings(), {'enabled': False, 'stop': 0.9, 'weight': 0.75})

    def test_face_lives_next_to_config(self):
        self.assertEqual(os.path.dirname(G.face_path()), self.root)

    def test_save_writes_face_and_config_block(self):
        html = self.enable(stop=0.8, weight=1.2)
        self.assertTrue(os.path.isfile(G.face_path()))
        with open(self.cfg.config_path, encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data['global_faceswap'], {'enabled': True, 'stop': 0.8, 'weight': 1.2})
        self.assertEqual(self.cfg.global_faceswap_config, data['global_faceswap'])
        self.assertIn('Active', html)
        face = G.load_face()
        self.assertEqual(face.shape, (32, 24, 3))
        self.assertEqual(int(face[0, 0, 0]), 200)

    def test_save_keeps_the_other_config_keys(self):
        with open(self.cfg.config_path, 'w', encoding='utf-8') as f:
            json.dump({'default_model': 'x.safetensors'}, f)
        self.enable()
        with open(self.cfg.config_path, encoding='utf-8') as f:
            data = json.load(f)
        self.assertEqual(data['default_model'], 'x.safetensors')
        self.assertIn('global_faceswap', data)

    def test_none_image_removes_the_face(self):
        self.enable()
        html = G.save(True, None, 0.9, 0.75)
        self.assertFalse(os.path.isfile(G.face_path()))
        self.assertIsNone(G.load_face())
        self.assertIn('no face', html)

    def test_rgba_and_grey_faces_are_normalised(self):
        G.save_face(np.zeros((8, 8, 4), np.uint8))
        self.assertEqual(G.load_face().shape, (8, 8, 3))
        G.save_face(np.zeros((8, 8), np.uint8))
        self.assertEqual(G.load_face().shape, (8, 8, 3))

    def test_settings_are_clamped(self):
        self.cfg.global_faceswap_config = {'enabled': 1, 'stop': 7, 'weight': 'oops'}
        self.assertEqual(G.settings(), {'enabled': True, 'stop': 1.0, 'weight': 0.75})

    def test_off_status(self):
        self.assertIn('Off', G.status_html())


class TestInject(World):
    def test_disabled_does_nothing(self):
        t = FakeTask()
        self.assertFalse(G.inject(t, log=self.logs.append))
        self.assertEqual(t.cn_tasks[FACE], [])

    def test_enabled_without_face_logs_and_skips(self):
        self.enable(with_face=False)
        t = FakeTask()
        self.assertFalse(G.inject(t, log=self.logs.append))
        self.assertIn('no face', self.logs[0])

    def test_text_to_image_turns_input_image_on_with_the_ip_tab(self):
        self.enable()
        t = FakeTask(tab='uov', input_image=False, others=(CANNY,))
        self.assertTrue(G.inject(t, log=self.logs.append))
        self.assertTrue(t.input_image_checkbox)
        self.assertEqual(t.current_tab, 'ip')
        self.assertEqual(len(t.cn_tasks[FACE]), 1)
        self.assertEqual(t.cn_tasks[CANNY], [], 'stale Image Prompt tasks are not revived')
        self.assertFalse(t.mixing_image_prompt_and_vary_upscale)
        self.assertFalse(t.mixing_image_prompt_and_inpaint)

    def test_ip_tab_just_appends(self):
        self.enable()
        t = FakeTask(tab='ip', others=(IP,))
        G.inject(t, log=self.logs.append)
        self.assertEqual(len(t.cn_tasks[IP]), 1)
        self.assertEqual(len(t.cn_tasks[FACE]), 1)
        self.assertFalse(t.mixing_image_prompt_and_vary_upscale)

    def test_uov_tab_sets_the_vary_flag_only(self):
        self.enable()
        t = FakeTask(tab='uov', others=(CANNY,))
        G.inject(t, log=self.logs.append)
        self.assertTrue(t.mixing_image_prompt_and_vary_upscale)
        self.assertFalse(t.mixing_image_prompt_and_inpaint)
        self.assertEqual(t.cn_tasks[CANNY], [])

    def test_inpaint_tab_sets_the_inpaint_flag_only(self):
        self.enable()
        t = FakeTask(tab='inpaint')
        G.inject(t, log=self.logs.append)
        self.assertTrue(t.mixing_image_prompt_and_inpaint)
        self.assertFalse(t.mixing_image_prompt_and_vary_upscale)

    def test_enhance_tab_reaches_the_cn_goal_through_the_vary_flag(self):
        self.enable()
        t = FakeTask(tab='enhance')
        G.inject(t, log=self.logs.append)
        self.assertTrue(t.mixing_image_prompt_and_vary_upscale)
        self.assertEqual(t.current_tab, 'enhance')

    def test_user_mixing_keeps_their_other_tasks(self):
        self.enable()
        t = FakeTask(tab='uov', mixing=(True, False), others=(CANNY,))
        G.inject(t, log=self.logs.append)
        self.assertEqual(len(t.cn_tasks[CANNY]), 1)

    def test_task_carries_stop_and_weight(self):
        self.enable(stop=0.6, weight=1.1)
        t = FakeTask(tab='ip')
        G.inject(t, log=self.logs.append)
        face, stop, weight = t.cn_tasks[FACE][0]
        self.assertEqual((stop, weight), (0.6, 1.1))
        self.assertEqual(face.shape, (32, 24, 3))

    def test_task_without_cn_tasks_is_ignored(self):
        self.enable()
        self.assertFalse(G.inject(object(), log=self.logs.append))


if __name__ == '__main__':
    unittest.main()
