"""custom-26 — protocole CLI de la famille crispz, sans GPU ni Fooocus lance.

  - validation des specs (codes 2 / 3, avertissements) : pur ;
  - route distante contre un faux Fooocus (Gradio 3 : POST /run/<nom>) ;
  - correspondance spec -> snapshot AsyncTask (upscale, variation, inpaint, refs,
    detail_faces, LoRA) sur la config reelle, sans charger de modele.

Run:  py -3.10 -m unittest tests.test_protocol -v
"""
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import numpy as np
from PIL import Image

sys.argv = sys.argv[:1]  # modules.config -> args_manager analyse sys.argv a l'import

import fooocus_protocol as P  # noqa: E402


def png(path, size=(64, 48), color=(200, 200, 200)):
    Image.new('RGB', size, color).save(path)
    return path


class TestSpecValidation(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='fczp_')
        self.img = png(os.path.join(self.dir, 'in.png'))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def code(self, spec, op):
        with self.assertRaises(P.ProtocolError) as cm:
            P.validate_spec(spec, op)
        return cm.exception.code

    def test_protocol_and_unsupported_ops_exit_3(self):
        self.assertEqual(self.code({'protocol': 2, 'prompt': 'x'}, 'gen'), 3)
        self.assertEqual(self.code({'protocol': 1, 'prompt': 'x'}, 'edit'), 3)

    def test_unknown_fields_and_count_are_warnings_not_errors(self):
        s, w = P.validate_spec({'protocol': 1, 'prompt': 'a cat', 'count': 4, 'qwen_only': 1}, 'gen')
        self.assertEqual(s['prompt'], 'a cat')
        self.assertTrue(any('count 4 forced to 1' in x for x in w))
        self.assertTrue(any('"qwen_only" ignored' in x for x in w))

    def test_prompt_lora_tags_are_merged_and_stripped_explicit_wins(self):
        s, _ = P.validate_spec({'protocol': 1, 'prompt': '@Lea runs <lora:ink:0.6>, rain <lora:grain>',
                                'loras': ['ink:0.9']}, 'gen')
        self.assertEqual(s['prompt'], '@Lea runs, rain')
        self.assertEqual(s['loras'], [('ink', 0.9), ('grain', None)])

    def test_bad_specs_exit_2(self):
        self.assertEqual(self.code({'protocol': 1, 'prompt': '<lora:ink:0.5>'}, 'gen'), 2)
        self.assertEqual(self.code({'protocol': 1, 'prompt': 'x', 'loras': ['ink:heavy']}, 'gen'), 2)
        self.assertEqual(self.code({'protocol': 1, 'prompt': 'x', 'refs': ['C:/nope.png']}, 'gen'), 2)
        self.assertEqual(self.code({'protocol': 1, 'prompt': ''}, 'gen'), 2)
        self.assertEqual(self.code({'protocol': 1, 'prompt': 'x', 'width': 1024}, 'gen'), 2)
        self.assertEqual(self.code({'protocol': 1, 'input': self.img, 'factor': 3}, 'upscale'), 2)
        self.assertEqual(self.code({'protocol': 1, 'factor': 2}, 'upscale'), 2)
        self.assertEqual(self.code({'protocol': 1, 'input': self.img, 'denoise': 1.5}, 'upscale'), 2)

    def test_an_all_black_mask_exits_2_and_a_size_mismatch_warns(self):
        black = png(os.path.join(self.dir, 'black.png'), color=(0, 0, 0))
        self.assertEqual(self.code({'protocol': 1, 'input': self.img, 'mask': black}, 'inpaint'), 2)
        white = png(os.path.join(self.dir, 'white.png'), size=(32, 24), color=(255, 255, 255))
        _, w = P.validate_spec({'protocol': 1, 'input': self.img, 'mask': white}, 'inpaint')
        self.assertTrue(any('resized' in x for x in w))

    def test_sizes_are_rounded_and_refs_capped(self):
        refs = [png(os.path.join(self.dir, f'r{i}.png')) for i in range(6)]
        s, w = P.validate_spec({'protocol': 1, 'prompt': 'x', 'width': 1030, 'height': 1344,
                                'refs': refs}, 'gen')
        self.assertEqual((s['width'], s['height'], len(s['refs'])), (1024, 1344, 4))
        self.assertEqual(len(w), 2)

    def test_relative_paths_are_resolved_from_the_caller(self):
        with mock.patch.dict(os.environ, {'CZP_CALLER_CWD': self.dir}):
            s, _ = P.validate_spec({'protocol': 1, 'input': 'in.png', 'factor': 1}, 'upscale')
        self.assertEqual(s['input'], self.img)


class FakeFooocus(BaseHTTPRequestHandler):
    tool = 'fooocus2026'
    last = None

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)))
        if self.path == '/run/cli_caps':
            out = json.dumps({'ok': True, 'tool': FakeFooocus.tool, 'protocol': 1})
        elif self.path == '/run/cli_gen':
            spec = json.loads(body['data'][0])
            FakeFooocus.last = spec
            if spec.get('prompt') == 'boom':
                out = json.dumps({'ok': False, 'error': 'refused', 'code': 2})
            else:
                out = json.dumps({'ok': True, 'tool': 'fooocus2026', 'route': 'remote',
                                  'images': ['C:/out/1.png'], 'seed_used': 7})
        else:
            self.send_response(404)
            self.end_headers()
            return
        b = json.dumps({'data': [out], 'duration': 0.1}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class TestRemoteRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(('127.0.0.1', 0), FakeFooocus)
        cls.url = f'http://127.0.0.1:{cls.httpd.server_address[1]}'
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='fczp_')
        FakeFooocus.tool = 'fooocus2026'

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def czp(self, *argv, spec=None):
        if spec is not None:
            path = os.path.join(self.dir, 'spec.json')
            with open(path, 'w', encoding='utf-8-sig') as f:  # BOM PowerShell accepte
                json.dump(spec, f)
            argv = argv + ('--spec', path)
        buf = io.StringIO()
        code = P.main(list(argv), out=buf)
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 1, 'toujours UNE ligne JSON')
        return code, json.loads(lines[0])

    def test_caps_through_the_instance(self):
        code, caps = self.czp('caps', '--remote', self.url)
        self.assertEqual((code, caps['tool'], caps['route']), (0, 'fooocus2026', 'remote'))

    def test_gen_is_routed_to_the_instance_with_absolute_paths(self):
        png(os.path.join(self.dir, 'ref.png'))
        with mock.patch.dict(os.environ, {'CZP_CALLER_CWD': self.dir}):
            code, res = self.czp('gen', '--remote', self.url,
                                 spec={'protocol': 1, 'prompt': 'a cat', 'refs': ['ref.png'], 'out_dir': 'out'})
        self.assertEqual((code, res['ok'], res['images']), (0, True, ['C:/out/1.png']))
        self.assertEqual(FakeFooocus.last['op'], 'gen')
        self.assertEqual(FakeFooocus.last['refs'], [os.path.join(self.dir, 'ref.png')])
        self.assertEqual(FakeFooocus.last['out_dir'], os.path.join(self.dir, 'out'))

    def test_a_remote_refusal_keeps_its_exit_code(self):
        code, res = self.czp('gen', '--remote', self.url, spec={'protocol': 1, 'prompt': 'boom'})
        self.assertEqual((code, res['ok']), (2, False))

    def test_another_app_on_the_port_is_never_taken_for_fooocus(self):
        FakeFooocus.tool = 'crispz-studio'
        code, res = self.czp('gen', '--remote', self.url, spec={'protocol': 1, 'prompt': 'a cat'})
        self.assertEqual(code, 4)
        self.assertIn('no Fooocus2026 instance', res['error'])

    def test_spec_errors_come_before_any_route(self):
        code, res = self.czp('gen', '--remote', 'http://127.0.0.1:9', spec={'protocol': 1, 'prompt': ''})
        self.assertEqual(code, 2)
        code, _ = self.czp('edit', '--remote', self.url, spec={'protocol': 1, 'prompt': 'x'})
        self.assertEqual(code, 3)
        code, _ = self.czp('gen', '--remote', self.url)
        self.assertEqual(code, 2)


class TestTaskArgsMapping(unittest.TestCase):
    """Spec -> snapshot, sur la config reelle, sans charger de modele."""

    @classmethod
    def setUpClass(cls):
        import modules.config as cfg
        import modules.flags as flags
        import modules.task_args as task_args
        cls.cfg, cls.flags, cls.T = cfg, flags, task_args

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='fczp_')
        self.img = png(os.path.join(self.dir, 'in.png'))

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def build(self, spec, op, route='local'):
        s, w = P.validate_spec(dict(spec, protocol=1), op)
        s['seed_used'] = 42
        args, info = P.build_task_args(s, route, w)
        _, idx = self.T.build(prompt='x', seed=1)
        return args, idx, info, w

    def test_gen_size_steps_refs_and_detail(self):
        with mock.patch.object(self.cfg, 'lora_filenames', ['styles/ink_v2.safetensors']):
            args, idx, info, _ = self.build({'prompt': 'a cat <lora:ink_v2:0.7>', 'width': 1024,
                                             'height': 1344, 'steps': 20, 'refs': [self.img],
                                             'detail_faces': True, 'detail_hands': True}, 'gen')
        self.assertEqual(args[idx['aspect']], '1024×1344')
        self.assertEqual((args[idx['seed']], args[idx['overwrite_step']]), (42, 20))
        self.assertEqual((args[idx['lora0_name']], args[idx['lora0_weight']]), ('styles/ink_v2.safetensors', 0.7))
        self.assertEqual(info, {'refs_used': 1, 'loras': ['styles/ink_v2.safetensors:0.7']})
        self.assertEqual((args[idx['current_tab']], args[idx['cn0_type']]), ('ip', self.flags.cn_ip))
        self.assertEqual(args[idx['cn0_image']].shape, (48, 64, 3))
        self.assertTrue(args[idx['enhance_checkbox']])
        self.assertEqual((args[idx['enh0_mask_prompt']], args[idx['enh1_mask_prompt']]), ('face', 'hand'))

    def test_an_unknown_lora_exits_2(self):
        with mock.patch.object(self.cfg, 'lora_filenames', ['a.safetensors']):
            with self.assertRaises(P.ProtocolError) as cm:
                self.build({'prompt': 'x', 'loras': ['nope:0.5']}, 'gen')
        self.assertEqual(cm.exception.code, 2)

    def test_model_is_refused_on_the_remote_route(self):
        args, idx, _, w = self.build({'prompt': 'x', 'model': 'juggernaut'}, 'gen', route='remote')
        self.assertEqual(args[idx['base_model']], self.cfg.default_base_model_name)
        self.assertTrue(any('refused on the remote route' in x for x in w))

    def test_upscale_factors(self):
        args, idx, _, _ = self.build({'input': self.img, 'factor': 1, 'denoise': 0.4}, 'upscale')
        self.assertEqual((args[idx['uov_method']], args[idx['overwrite_vary_strength']]),
                         (self.flags.subtle_variation, 0.4))
        args, idx, _, _ = self.build({'input': self.img, 'factor': 2, 'denoise': 0}, 'upscale')
        self.assertEqual(args[idx['uov_method']], self.flags.upscale_fast)
        args, idx, _, _ = self.build({'input': self.img, 'factor': 1.5, 'denoise': 0.3}, 'upscale')
        self.assertEqual((args[idx['uov_method']], args[idx['overwrite_upscale_strength']]),
                         (self.flags.upscale_15, 0.3))
        self.assertEqual(args[idx['uov_input_image']].shape, (48, 64, 3))

    def test_inpaint_uses_the_local_prompt_and_the_white_mask(self):
        mask = os.path.join(self.dir, 'mask.png')
        m = Image.new('L', (64, 48), 0)
        m.paste(255, (10, 10, 30, 30))
        m.save(mask)
        args, idx, _, _ = self.build({'input': self.img, 'mask': mask, 'prompt': 'an umbrella',
                                      'denoise': 0.8}, 'inpaint')
        self.assertEqual((args[idx['current_tab']], args[idx['prompt']], args[idx['inpaint_additional_prompt']]),
                         ('inpaint', '', 'an umbrella'))
        self.assertEqual(args[idx['inpaint_strength']], 0.8)
        pair = args[idx['inpaint_input_image']]
        self.assertEqual(pair['mask'].shape, (48, 64, 3))
        self.assertEqual(int(pair['mask'][20, 20, 0]), 255)
        self.assertEqual(int(pair['mask'][0, 0, 0]), 0)


class Cp1252Stream(unittest.TestCase):
    """The JSON line is written in UTF-8 even when the stream defaults to
    cp1252 (Windows console / pipe): an accented or emoji name must not end
    the command with UnicodeEncodeError."""

    def test_json_line_survives_a_cp1252_stream(self):
        raw = io.BytesIO()
        out = io.TextIOWrapper(raw, encoding='cp1252', newline='\n')
        name = 'bidule\u00e9\U0001f98b'
        code = P.main([name], out=out)
        out.flush()
        self.assertEqual(code, 3)
        line = raw.getvalue().decode('utf-8')
        self.assertIn(name, line)
        self.assertFalse(json.loads(line)['ok'])


if __name__ == '__main__':
    unittest.main()
