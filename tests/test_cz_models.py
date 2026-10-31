"""crispz-sdxl: the SDXL-only lists of cli_caps, on crafted safetensors headers
(a header is JSON after an 8-byte length, no weights needed).

Run:  python -m pytest tests/test_cz_models.py -q
"""
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from unittest import mock

sys.argv = sys.argv[:1]

import cz_models  # noqa: E402


def safetensors(folder, name, keys):
    header = json.dumps({k: {'dtype': 'F16', 'shape': [1], 'data_offsets': [0, 2]} for k in keys}).encode()
    with open(os.path.join(folder, name), 'wb') as f:
        f.write(struct.pack('<Q', len(header)) + header + b'\0\0' * len(keys))
    return name


class TestSdxlLists(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='czm_')
        self.cache = os.path.join(self.dir, 'cache.json')
        self.patch = mock.patch.object(cz_models, '_cache_path', return_value=self.cache)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_checkpoints_and_loras_are_classified_by_header(self):
        d = self.dir
        models = [
            safetensors(d, 'xl.safetensors', ['model.diffusion_model.input_blocks.0.0.weight',
                                              'model.diffusion_model.label_emb.0.0.weight']),
            safetensors(d, 'sd15.safetensors', ['model.diffusion_model.input_blocks.0.0.weight']),
            safetensors(d, 'svd.safetensors', ['model.diffusion_model.time_stack.0.weight']),
            safetensors(d, 'clip.safetensors', ['text_model.encoder.layers.0.weight']),
        ]
        with open(os.path.join(d, 'old.ckpt'), 'wb') as f:
            f.write(b'x')
        models.append('old.ckpt')
        loras = [
            safetensors(d, 'kohya_xl.safetensors',
                        ['lora_unet_input_blocks_4_1_transformer_blocks_1_attn1_to_k.lora_down.weight']),
            safetensors(d, 'te2_xl.safetensors', ['lora_te2_text_model_encoder_layers_0_mlp_fc1.lora_down.weight']),
            safetensors(d, 'diffusers_xl.safetensors',
                        ['unet.down_blocks.1.attentions.0.transformer_blocks.1.attn1.to_k.lora_A.weight']),
            safetensors(d, 'sd15.safetensors.lora.safetensors',
                        ['lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_k.lora_down.weight',
                         'lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight']),
            safetensors(d, 'flux.safetensors', ['lora_transformer_double_blocks_0_img_attn_proj.lora_down.weight']),
            safetensors(d, 'qwen.safetensors', ['transformer.transformer_blocks.0.attn.to_k.lora_A.weight']),
            safetensors(d, 'sd3.safetensors', ['lora_transformer_joint_blocks_0_x_block_attn_proj.lora_down.weight']),
            safetensors(d, 'misplaced_checkpoint.safetensors', ['model.diffusion_model.input_blocks.0.0.weight']),
            'missing.safetensors',
        ]
        kept_m, kept_l, report = cz_models.sdxl_lists(models, loras, [d], [d])
        self.assertEqual(kept_m, ['xl.safetensors', 'old.ckpt'])
        self.assertEqual(kept_l, ['kohya_xl.safetensors', 'te2_xl.safetensors', 'diffusers_xl.safetensors',
                                  'missing.safetensors'])
        self.assertEqual(report, {'checkpoints': {'sd': 1, 'svd': 1, 'other': 1},
                                  'loras': {'sd': 1, 'flux': 1, 'dit': 1, 'sd3': 1, 'not_a_lora': 1}})
        with open(self.cache, encoding='utf-8') as f:
            cache = json.load(f)
        self.assertEqual(sum(1 for k in cache if k.startswith('lora:')), 8)
        # second pass: the cache answers, no header is read
        with mock.patch.object(cz_models, 'read_keys', side_effect=AssertionError('header read again')):
            with mock.patch.object(cz_models, 'checkpoint_arch', side_effect=AssertionError('header read again')):
                kept_m2, kept_l2, _ = cz_models.sdxl_lists(models, loras, [d], [d])
        self.assertEqual((kept_m2, kept_l2), (kept_m, kept_l))

    def test_an_empty_library_reports_nothing(self):
        self.assertEqual(cz_models.sdxl_lists([], [], [self.dir], [self.dir]), ([], [], {}))
        self.assertFalse(os.path.exists(self.cache))


if __name__ == '__main__':
    unittest.main()
