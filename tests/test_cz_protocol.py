"""crispz-sdxl lot 1: the CLI protocol v1 endpoints on CPU, with the fake engine.

Covers cli_caps (identity, capabilities computed from what runs, model lists)
and a cli_gen round trip through FakeWorker: the spec goes through the
custom-26 validator and mapping (fooocus_protocol.build_task_args), the fake
engine decodes the AsyncTask snapshot by name and writes an image in the
family layout (out/YYYY-MM-DD/{date}_{tag}_seed{seed}_{w}x{h}.png with the
crispz chunk and the .json sidecar). No torch, no gradio, no model.

Run:  python -m pytest tests/test_cz_protocol.py -q
      python -m unittest tests.test_cz_protocol -v
"""
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

sys.argv = sys.argv[:1]  # modules.config -> args_manager parses sys.argv at import

import cz_core  # noqa: E402
import cz_imageio  # noqa: E402
import cz_pipeline  # noqa: E402
import cz_protocol  # noqa: E402
import modules.config as cfg  # noqa: E402

MODELS = ['sdxl/juggernautXL_v9.safetensors', 'sdxl/realvisxlV40.safetensors']
LORAS = ['styles/ink_v2.safetensors', 'add_detail_xl.safetensors']


class FakeEngineCase(unittest.TestCase):
    """A temporary out/ and the fake engine on, for every test of the class."""

    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='crispz_sdxl_')
        self.env = mock.patch.dict(os.environ, {'CRISPZ_FAKE_ENGINE': '1', 'CRISPZ_OUT_DIR': self.dir,
                                                'CRISPZ_CONFIG': os.path.join(self.dir, 'no-config.txt')})
        self.env.start()
        self.lists = mock.patch.multiple(cfg, model_filenames=list(MODELS), lora_filenames=list(LORAS))
        self.lists.start()

    def tearDown(self):
        self.lists.stop()
        self.env.stop()
        shutil.rmtree(self.dir, ignore_errors=True)

    @staticmethod
    def call(handler, spec):
        out = handler(json.dumps(spec) if isinstance(spec, dict) else spec)
        assert isinstance(out, str), 'an endpoint answers a JSON string'
        return json.loads(out)


class TestCaps(FakeEngineCase):
    def test_identity_and_engine(self):
        caps = self.call(cz_protocol.cli_caps, '{}')
        self.assertEqual((caps['ok'], caps['protocol'], caps['tool'], caps['version']),
                         (True, 1, 'crispz-sdxl', '0.1.0'))
        self.assertEqual(caps['engine']['tool'], 'fooocus2026')
        self.assertTrue(caps['engine']['fake'])

    def test_models_and_loras_come_from_the_fooocus_lists(self):
        caps = self.call(cz_protocol.cli_caps, '{}')
        self.assertEqual((caps['models'], caps['loras']), (MODELS, LORAS))
        self.assertFalse(caps['model_loaded'])

    def test_supports_never_claims_more_than_the_pipeline_runs(self):
        caps = self.call(cz_protocol.cli_caps, '{}')
        s = caps['supports']
        self.assertEqual(s['max_refs'], min(cfg.default_controlnet_image_count, 4))
        for op in cz_pipeline.IMPLEMENTED_OPS:
            self.assertIn(op, caps['ops'])
        self.assertEqual(s['edit'], 'edit' in cz_pipeline.IMPLEMENTED_OPS)
        self.assertEqual(s['faces'], 'faces' in cz_pipeline.IMPLEMENTED_OPS)
        self.assertEqual(set(caps['planned']), {'edit', 'faces'} - set(cz_pipeline.IMPLEMENTED_OPS))
        self.assertTrue(s['loras'] and s['seed'] and s['negative'] and s['arbitrary_size'])

    def test_the_four_family_endpoints_exist(self):
        self.assertEqual(set(cz_protocol.endpoints()), {'cli_caps', 'cli_gen', 'cli_edit', 'cli_faces'})


class TestGenRoundTrip(FakeEngineCase):
    SPEC = {'protocol': 1, 'op': 'gen', 'prompt': 'a red cat on a roof <lora:ink_v2:0.7>',
            'negative': 'blurry', 'width': 256, 'height': 320, 'seed': 7, 'steps': 4,
            'model': 'juggernaut'}

    def test_gen_writes_a_family_image_through_the_fake_engine(self):
        res = self.call(cz_protocol.cli_gen, self.SPEC)
        self.assertTrue(res['ok'], res)
        self.assertEqual((res['tool'], res['version'], res['seed_used']), ('crispz-sdxl', '0.1.0', 7))
        self.assertTrue(res['engine']['fake'])
        self.assertTrue(any('FAKE engine' in w for w in res['warnings']), 'the degradation is announced')
        self.assertEqual(res['loras'], ['styles/ink_v2.safetensors:0.7'])

        self.assertEqual(len(res['images']), 1)
        path = res['images'][0]
        self.assertTrue(os.path.isfile(path), path)
        day = cz_core.today()
        self.assertEqual(os.path.dirname(path), os.path.join(self.dir, day))
        self.assertEqual(os.path.basename(path), f'{day}_sdxl_seed7_256x320.png')
        with Image.open(path) as im:
            self.assertEqual(im.size, (256, 320))
            self.assertIn('parameters', im.text)

        meta = cz_imageio.read_meta(path)
        self.assertEqual((meta['tool'], meta['engine'], meta['seed'], meta['steps']), ('crispz-sdxl', 'fake', 7, 4))
        self.assertEqual(meta['prompt'], 'a red cat on a roof')  # the LoRA tag is stripped
        self.assertEqual(meta['model'], 'sdxl/juggernautXL_v9.safetensors')  # resolved by partial name
        sidecar = os.path.splitext(path)[0] + '.json'
        with open(sidecar, encoding='utf-8') as f:
            self.assertEqual(json.load(f), meta)

    def test_a_second_render_with_the_same_seed_never_overwrites(self):
        first = self.call(cz_protocol.cli_gen, self.SPEC)['images'][0]
        second = self.call(cz_protocol.cli_gen, self.SPEC)['images'][0]
        self.assertNotEqual(first, second)
        self.assertTrue(re.search(r'_seed7_256x320_2\.png$', second), second)

    def test_out_dir_of_the_spec_receives_a_copy(self):
        target = os.path.join(self.dir, 'dataset', 'lea')
        res = self.call(cz_protocol.cli_gen, dict(self.SPEC, out_dir=target))
        self.assertTrue(res['ok'], res)
        self.assertEqual(os.path.dirname(res['images'][0]), target)
        self.assertTrue(os.path.isfile(res['images'][0]))

    def test_a_random_seed_is_reported(self):
        res = self.call(cz_protocol.cli_gen, dict(self.SPEC, seed=-1))
        self.assertTrue(res['ok'], res)
        self.assertIn(f'seed{res["seed_used"]}_', os.path.basename(res['images'][0]))

    def test_upscale_and_inpaint_ride_cli_gen(self):
        src = os.path.join(self.dir, 'in.png')
        Image.new('RGB', (64, 48), (10, 20, 30)).save(src)
        res = self.call(cz_protocol.cli_gen, {'protocol': 1, 'op': 'upscale', 'input': src, 'factor': 2})
        self.assertTrue(res['ok'], res)
        with Image.open(res['images'][0]) as im:
            self.assertEqual(im.size, (128, 96))
        mask = os.path.join(self.dir, 'mask.png')
        m = Image.new('L', (64, 48), 0)
        m.paste(255, (0, 0, 32, 48))
        m.save(mask)
        res = self.call(cz_protocol.cli_gen, {'protocol': 1, 'op': 'inpaint', 'input': src, 'mask': mask,
                                              'prompt': 'an umbrella'})
        self.assertTrue(res['ok'], res)
        with Image.open(res['images'][0]) as im:
            self.assertEqual(im.size, (64, 48))
            self.assertEqual(im.getpixel((60, 40)), (10, 20, 30))  # outside the mask: untouched


class TestRefusals(FakeEngineCase):
    def test_bad_specs_answer_code_2_in_json(self):
        self.assertEqual(self.call(cz_protocol.cli_gen, 'not json')['code'], 2)
        self.assertEqual(self.call(cz_protocol.cli_gen, '[1, 2]')['code'], 2)
        self.assertEqual(self.call(cz_protocol.cli_gen, {'protocol': 1, 'prompt': ''})['code'], 2)
        res = self.call(cz_protocol.cli_gen, {'protocol': 1, 'prompt': 'x', 'model': 'nope'})
        self.assertEqual((res['ok'], res['code']), (False, 2))
        self.assertIn('checkpoint not found', res['error'])

    def test_foreign_protocol_and_wrong_endpoint_answer_code_3(self):
        self.assertEqual(self.call(cz_protocol.cli_gen, {'protocol': 2, 'prompt': 'x'})['code'], 3)
        self.assertEqual(self.call(cz_protocol.cli_gen, {'protocol': 1, 'op': 'edit', 'prompt': 'x'})['code'], 3)

    def test_the_dedicated_endpoints_refuse_a_foreign_op(self):
        self.assertEqual(self.call(cz_protocol.cli_edit, {'protocol': 1, 'op': 'gen', 'prompt': 'x'})['code'], 2)
        self.assertEqual(self.call(cz_protocol.cli_edit, {'protocol': 1, 'op': 'faces', 'input': 'x.png'})['code'], 3)
        self.assertEqual(self.call(cz_protocol.cli_faces, {'protocol': 1, 'op': 'edit', 'input': 'x.png'})['code'], 3)
        self.assertEqual(self.call(cz_protocol.cli_faces, {'protocol': 1, 'op': 'faces'})['code'], 2)

    @unittest.skipIf('torch' in sys.modules or __import__('importlib').util.find_spec('torch'),
                     'torch is installed here: the real engine would import')
    def test_without_the_fake_engine_the_missing_engine_is_explained(self):
        with mock.patch.dict(os.environ, {'CRISPZ_FAKE_ENGINE': '0'}):
            res = self.call(cz_protocol.cli_gen, {'protocol': 1, 'prompt': 'x'})
        self.assertEqual((res['ok'], res['code']), (False, 1))
        self.assertIn('CRISPZ_FAKE_ENGINE=1', res['error'])


class TestEdit(FakeEngineCase):
    def setUp(self):
        super().setUp()
        self.src = os.path.join(self.dir, 'in.png')
        Image.new('RGB', (64, 48), (10, 20, 30)).save(self.src)

    def test_edit_without_a_mask_is_a_strong_variation_of_the_input(self):
        spec = {'protocol': 1, 'op': 'edit', 'prompt': 'make it night', 'input': self.src, 'seed': 3,
                'width': 1024, 'height': 1024, 'denoise': 0.4, 'refs': [self.src]}
        res = self.call(cz_protocol.cli_edit, spec)
        self.assertTrue(res['ok'], res)
        self.assertEqual((res['edit_mode'], res['denoise'], res['refs_used'], res['seed_used']), ('img2img', 0.4, 1, 3))
        self.assertTrue(any('width/height ignored' in w for w in res['warnings']))
        with Image.open(res['images'][0]) as im:
            self.assertEqual(im.size, (64, 48))  # the input size is kept
        self.assertEqual(cz_imageio.read_meta(res['images'][0])['op'], 'edit')

    def test_edit_with_a_mask_is_an_inpaint_with_the_main_prompt(self):
        mask = os.path.join(self.dir, 'mask.png')
        m = Image.new('L', (64, 48), 0)
        m.paste(255, (0, 0, 32, 48))
        m.save(mask)
        res = self.call(cz_protocol.cli_edit, {'protocol': 1, 'op': 'edit', 'prompt': 'a window', 'input': self.src,
                                               'mask': mask})
        self.assertTrue(res['ok'], res)
        self.assertEqual((res['edit_mode'], res['denoise']), ('inpaint', cz_pipeline.EDIT_DEFAULT_DENOISE))
        with Image.open(res['images'][0]) as im:
            self.assertEqual(im.getpixel((60, 40)), (10, 20, 30))

    def test_edit_mapping_on_the_snapshot(self):
        import modules.flags as flags
        import modules.task_args as task_args
        s, w = cz_pipeline.validate_edit_spec({'protocol': 1, 'prompt': 'x', 'input': self.src, 'refs': [self.src]})
        s['seed_used'] = 1
        args, info = cz_pipeline.build_edit_args(s, 'local', w)
        _, idx = task_args.build(prompt='x', seed=1)
        self.assertEqual((args[idx['current_tab']], args[idx['uov_method']]), ('uov', flags.strong_variation))
        self.assertEqual(args[idx['overwrite_vary_strength']], cz_pipeline.EDIT_DEFAULT_DENOISE)
        self.assertTrue(args[idx['input_image_checkbox']] and args[idx['mixing_image_prompt_and_vary_upscale']])
        self.assertEqual(args[idx['uov_input_image']].shape, (48, 64, 3))

    def test_edit_needs_an_input_and_a_prompt(self):
        self.assertEqual(self.call(cz_protocol.cli_edit, {'protocol': 1, 'op': 'edit', 'prompt': 'x'})['code'], 2)
        self.assertEqual(self.call(cz_protocol.cli_edit, {'protocol': 1, 'op': 'edit', 'input': self.src})['code'], 2)


class TestFaces(FakeEngineCase):
    def test_faces_answers_boxes_mouth_and_a_normed_512_embedding(self):
        src = os.path.join(self.dir, 'face.png')
        Image.new('RGB', (200, 100), (90, 60, 50)).save(src)
        res = self.call(cz_protocol.cli_faces, {'protocol': 1, 'op': 'faces', 'input': src})
        self.assertTrue(res['ok'], res)
        self.assertEqual((res['tool'], res['count'], res['width'], res['height']), ('crispz-sdxl', 1, 200, 100))
        face = res['faces'][0]
        self.assertEqual(face['box'], [50, 25, 150, 75])
        self.assertEqual(len(face['embedding']), 512)
        self.assertAlmostEqual(sum(v * v for v in face['embedding']), 1.0, places=5)
        self.assertTrue(any('FAKE' in w for w in res['warnings']))
        again = self.call(cz_protocol.cli_faces, {'protocol': 1, 'op': 'faces', 'input': src})
        self.assertEqual(again['faces'][0]['embedding'], face['embedding'])

    def test_caps_say_faces_only_when_a_detector_exists(self):
        self.assertTrue(self.call(cz_protocol.cli_caps, '{}')['supports']['faces'])
        with mock.patch.dict(os.environ, {'CRISPZ_FAKE_ENGINE': '0'}):
            with mock.patch('cz_face.importlib.util.find_spec', return_value=None):
                caps = self.call(cz_protocol.cli_caps, '{}')
                self.assertFalse(caps['supports']['faces'])
                self.assertNotIn('faces', caps['ops'])
                res = self.call(cz_protocol.cli_faces, {'protocol': 1, 'op': 'faces', 'input': __file__})
                self.assertEqual(res['code'], 1)
                self.assertIn('insightface', res['error'])


class TestRemoteRoute(FakeEngineCase):
    def test_bound_in_the_app_the_model_of_the_spec_is_refused_unless_allowed(self):
        with mock.patch.object(cz_protocol, 'ROUTE', 'remote'):
            res = self.call(cz_protocol.cli_gen, {'protocol': 1, 'prompt': 'x', 'model': 'juggernaut', 'seed': 1})
            self.assertTrue(res['ok'], res)
            self.assertTrue(any('refused on the remote route' in w for w in res['warnings']))
            self.assertEqual(cz_imageio.read_meta(res['images'][0])['model'], cfg.default_base_model_name)
            allowed = dict(cz_core.DEFAULTS, allow_model_switch=True)
            with mock.patch.object(cz_core, 'load_config', return_value=allowed):
                res = self.call(cz_protocol.cli_gen, {'protocol': 1, 'prompt': 'x', 'model': 'juggernaut', 'seed': 1})
            self.assertEqual(cz_imageio.read_meta(res['images'][0])['model'], MODELS[0])

    def test_bind_gradio_wires_the_four_endpoints_out_of_the_queue(self):
        bound = []

        class Widget:
            def __init__(self, **kw):
                pass

            def click(self, fn, **kw):
                bound.append((kw['api_name'], fn, kw['queue']))

        class Row(Widget):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        gr = mock.Mock(Row=Row, Textbox=Widget, Button=Widget)
        with mock.patch.object(cz_protocol, 'ROUTE', 'local'):
            buttons = cz_protocol.bind_gradio(gr)
            self.assertEqual(cz_protocol.ROUTE, 'remote')
        self.assertEqual([b[0] for b in bound], ['cli_caps', 'cli_gen', 'cli_edit', 'cli_faces'])
        self.assertEqual(set(buttons), set(cz_protocol.endpoints()))
        self.assertTrue(all(b[1] is cz_protocol.endpoints()[b[0]] and b[2] is False for b in bound))


class TestRealOutputImport(FakeEngineCase):
    def test_a_fooocus_file_becomes_a_family_file_and_keeps_its_a1111_chunk(self):
        from PIL.PngImagePlugin import PngInfo
        fooocus_dir = os.path.join(self.dir, 'outputs', cz_core.today())
        os.makedirs(fooocus_dir)
        src = os.path.join(fooocus_dir, '2026-10-31_10-00-00_1234.png')
        info = PngInfo()
        info.add_text('parameters', 'a cat\nSteps: 30, Seed: 5')
        info.add_text('fooocus_scheme', 'a1111')
        Image.new('RGB', (64, 32)).save(src, pnginfo=info)
        result = {'images': [src], 'seed_used': 5, 'loras': [], 'refs_used': 0}
        spec = {'prompt': 'a cat', 'steps': 30, 'out_dir': os.path.join(self.dir, 'ds')}
        res = cz_pipeline._import_outputs(result, spec, 'gen', fake=False)
        path = res['images'][0]
        self.assertEqual(os.path.dirname(path), os.path.join(self.dir, 'ds'))
        self.assertEqual(os.path.basename(path), f'{cz_core.today()}_sdxl_seed5_64x32.png')
        family = os.path.join(self.dir, cz_core.today(), os.path.basename(path))
        self.assertTrue(os.path.isfile(family) and os.path.isfile(src), 'out/ copy and Fooocus file both stay')
        with Image.open(path) as im:
            self.assertEqual(im.text['parameters'], 'a cat\nSteps: 30, Seed: 5')
        meta = cz_imageio.read_meta(path)
        self.assertEqual((meta['seed'], meta['width'], meta['source'], meta['op']), (5, 64, src, 'gen'))
        self.assertTrue(os.path.isfile(os.path.splitext(path)[0] + '.json'))


class TestCoreConfig(unittest.TestCase):
    def test_defaults_then_config_block_then_environment(self):
        d = tempfile.mkdtemp(prefix='crispz_cfg_')
        try:
            path = os.path.join(d, 'config.txt')
            with mock.patch.dict(os.environ, {'CRISPZ_CONFIG': path}, clear=False):
                os.environ.pop('CRISPZ_OUT_DIR', None)
                os.environ.pop('CRISPZ_FAKE_ENGINE', None)
                c = cz_core.load_config()
                self.assertEqual((c['tag'], c['port'], c['fake_engine']), ('sdxl', 7865, False))
                self.assertEqual(cz_core.out_dir(c), os.path.join(cz_core.ROOT, 'out'))
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump({'crispz': {'tag': 'xl', 'out_dir': d, 'fake_engine': True, 'port': 'x'}}, f)
                c = cz_core.load_config()
                self.assertEqual((c['tag'], c['port'], c['fake_engine']), ('xl', 7865, True))
                self.assertTrue(cz_core.fake_engine_enabled(c))
                self.assertEqual(cz_core.out_dir(c), d)
                with mock.patch.dict(os.environ, {'CRISPZ_FAKE_ENGINE': '0', 'CRISPZ_OUT_DIR': os.path.join(d, 'e')}):
                    self.assertFalse(cz_core.fake_engine_enabled(c))
                    self.assertEqual(cz_core.out_dir(c), os.path.join(d, 'e'))
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
