"""custom-39 — Batch input: one queue job per image of a folder (stdlib + numpy/PIL).

Run:  py -3.10 -m unittest tests.test_batch_input -v
"""
import json
import os
import random
import shutil
import sys
import tempfile
import types
import unittest

import numpy as np
from PIL import Image

from modules import batch_input as B
from modules.job_queue import encode_value, decode_value

# ctrls snapshot after the currentTask pop: only the slots the batch touches matter
IDX = {'input_image_checkbox': 0, 'current_tab': 1, 'uov_method': 2, 'uov_image': 3,
       'enhance_image': 4, 'enhance_checkbox': 5, 'seed': 6, 'base_model': 7}


def snapshot(tab='uov', method='Upscale (2x)', enhance=False, seed=42):
    return [False, tab, method, None, None, enhance, seed, 'juggernaut.safetensors', 'other', 3.5]


class World(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix='batch-')
        self.cfg = types.ModuleType('modules.config')
        self.settings = {}
        self.cfg.batch_input_setting = lambda k, d=None: self.settings.get(k, d)
        self._old = sys.modules.get('modules.config')
        sys.modules['modules.config'] = self.cfg

    def tearDown(self):
        if self._old is None:
            sys.modules.pop('modules.config', None)
        else:
            sys.modules['modules.config'] = self._old
        shutil.rmtree(self.root, ignore_errors=True)

    def png(self, name, size=(8, 6), color=(10, 20, 30), folder=None):
        path = os.path.join(folder or self.root, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.new('RGB', size, color).save(path)
        return path


class TestScan(World):
    def test_natural_order_and_image_filter(self):
        for n in ('img10.png', 'img2.jpg', 'img1.webp', 'notes.txt', 'IMG3.JPEG'):
            if n.endswith('.txt'):
                open(os.path.join(self.root, n), 'w').close()
            else:
                self.png(n)
        names = [os.path.basename(p) for p in B.scan_folder(self.root)]
        self.assertEqual(names, ['img1.webp', 'img2.jpg', 'IMG3.JPEG', 'img10.png'])

    def test_recursive_and_pattern(self):
        self.png('a.png')
        self.png('sub/b.png')
        self.png('sub/c.jpg')
        self.assertEqual(len(B.scan_folder(self.root)), 1)
        self.assertEqual(len(B.scan_folder(self.root, recursive=True)), 3)
        got = [os.path.basename(p) for p in B.scan_folder(self.root, recursive=True, pattern='*.png, c*')]
        self.assertEqual(got, ['a.png', 'b.png', 'c.jpg'])

    def test_check_folder_messages(self):
        ok, msg, _ = B.check_folder('')
        self.assertFalse(ok)
        self.assertIn('Type the path', msg)
        ok, msg, _ = B.check_folder(os.path.join(self.root, 'nope'))
        self.assertFalse(ok)
        self.assertIn('Not a folder', msg)
        ok, msg, path = B.check_folder(f'"{self.root}"')
        self.assertTrue(ok, msg)
        self.assertEqual(path, os.path.abspath(self.root))

    def test_allowed_roots(self):
        inside = os.path.join(self.root, 'ok')
        os.makedirs(inside)
        other = tempfile.mkdtemp(prefix='batch-other-')
        try:
            self.assertTrue(B.folder_allowed(other), 'no root configured: any folder')
            self.settings['allowed_roots'] = [self.root]
            self.assertTrue(B.folder_allowed(inside))
            self.assertFalse(B.folder_allowed(other))
            ok, msg, _ = B.check_folder(other)
            self.assertFalse(ok)
            self.assertIn('allowed_roots', msg)
            self.settings['allowed_roots'] = f'{self.root};{other}'
            self.assertTrue(B.folder_allowed(other), 'a ; separated string works too')
        finally:
            shutil.rmtree(other, ignore_errors=True)

    def test_scan_report_and_summary(self):
        for i in range(7):
            self.png(f'p{i}.png')
        report = B.scan_report(self.root)
        self.assertIn('7 image(s)', report)
        self.assertIn('(+2 more)', report)
        self.assertIn('No image found', B.scan_report(self.root, pattern='*.gif'))

    def test_collect_mixes_uploads_and_folder(self):
        self.png('f1.png')
        up = self.png('upload.png', folder=tempfile.mkdtemp(prefix='batch-up-'))
        entries, msg = B.collect(self.root, False, '', [types.SimpleNamespace(name=up)])
        self.assertEqual(msg, '')
        self.assertEqual([(os.path.basename(p), e) for p, e in entries], [('upload.png', True), ('f1.png', False)])
        entries, msg = B.collect('', False, '', None)
        self.assertEqual(entries, [])
        self.assertIn('Nothing to queue', msg)


class TestJobs(World):
    def test_one_job_per_image_with_lazy_refs_and_seeds(self):
        a = self.png('a.png')
        b = self.png('b.png')
        jobs, msg = B.build_jobs(snapshot(), [(a, False), (b, False)], IDX, keep_name=True,
                                 random_seed=True, rng=random.Random(1))
        self.assertEqual(msg, '')
        self.assertEqual(len(jobs), 2)
        (args1, label1, meta1), (args2, _, meta2) = jobs
        self.assertTrue(args1[IDX['input_image_checkbox']], 'Input Image forced on')
        self.assertTrue(B.is_file_ref(args1[IDX['uov_image']]))
        self.assertEqual(args1[IDX['uov_image']][B.FILE_KEY], os.path.abspath(a))
        self.assertNotEqual(args1[IDX['seed']], args2[IDX['seed']], 'random: one seed per image')
        self.assertEqual(meta1['batch']['stem'], 'a_upscaled')
        self.assertEqual((meta1['batch']['index'], meta1['batch']['total']), (1, 2))
        self.assertIn('a.png | Upscale (2x) | juggernaut | seed', label1)
        self.assertEqual(meta2['batch']['source'], os.path.abspath(b))

    def test_fixed_seed_applies_to_the_whole_batch(self):
        a = self.png('a.png')
        jobs, _ = B.build_jobs(snapshot(seed=7), [(a, False), (a, False)], IDX, random_seed=False)
        self.assertEqual([j[0][IDX['seed']] for j in jobs], [7, 7])

    def test_keep_name_off_gives_no_stem(self):
        a = self.png('a.png')
        jobs, _ = B.build_jobs(snapshot(), [(a, False)], IDX, keep_name=False)
        self.assertIsNone(jobs[0][2]['batch']['stem'])
        self.assertIsNone(B.output_stem_of(jobs[0][2]))
        self.assertIsNone(B.output_stem_of(None))
        self.assertIsNone(B.output_stem_of({'group': 'xyz'}))

    def test_enhance_tab_uses_its_slot(self):
        a = self.png('a.png')
        jobs, msg = B.build_jobs(snapshot(tab='enhance', enhance=True), [(a, False)], IDX)
        self.assertEqual(msg, '')
        args, label, meta = jobs[0]
        self.assertTrue(B.is_file_ref(args[IDX['enhance_image']]))
        self.assertIsNone(args[IDX['uov_image']])
        self.assertEqual(meta['batch']['stem'], 'a_enhanced')
        self.assertIn('| Enhance |', label)

    def test_refusals(self):
        a = self.png('a.png')
        for snap, word in ((snapshot(tab='ip'), 'select one of them'),
                           (snapshot(tab='enhance', enhance=False), 'Enhance checkbox'),
                           (snapshot(method='Disabled'), 'pick a method')):
            jobs, msg = B.build_jobs(snap, [(a, False)], IDX)
            self.assertEqual(jobs, [])
            self.assertIn(word, msg)

    def test_upload_is_decoded_at_once(self):
        a = self.png('a.png', size=(5, 4), color=(1, 2, 3))
        jobs, _ = B.build_jobs(snapshot(), [(a, True)], IDX)
        img = jobs[0][0][IDX['uov_image']]
        self.assertIsInstance(img, np.ndarray)
        self.assertEqual(img.shape, (4, 5, 3))
        self.assertEqual(tuple(img[0, 0]), (1, 2, 3))

    def test_vary_suffix(self):
        a = self.png('a.png')
        jobs, _ = B.build_jobs(snapshot(method='Vary (Strong)'), [(a, False)], IDX)
        self.assertEqual(jobs[0][2]['batch']['stem'], 'a_vary')


class TestMaterialize(World):
    def test_refs_survive_the_queue_codec_and_decode_on_demand(self):
        a = self.png('a.png', size=(6, 3), color=(9, 8, 7))
        jobs, _ = B.build_jobs(snapshot(), [(a, False)], IDX)
        args = jobs[0][0]
        used = set()
        enc = [encode_value(v, self.root, used) for v in args]
        back = [decode_value(v, self.root) for v in json.loads(json.dumps(enc))]
        self.assertEqual(used, set(), 'a lazy reference writes no asset file')
        self.assertTrue(B.is_file_ref(back[IDX['uov_image']]))
        out, missing = B.materialize(back)
        self.assertEqual(missing, '')
        self.assertEqual(out[IDX['uov_image']].shape, (3, 6, 3))
        self.assertEqual(tuple(out[IDX['uov_image']][0, 0]), (9, 8, 7))

    def test_missing_file_is_reported_not_raised(self):
        args = snapshot()
        args[IDX['uov_image']] = B.file_ref(os.path.join(self.root, 'gone.png'))
        out, missing = B.materialize(args)
        self.assertIn('source file missing', missing)
        self.assertTrue(B.is_file_ref(out[IDX['uov_image']]), 'args untouched when skipping')

    def test_unreadable_file_is_reported(self):
        bad = os.path.join(self.root, 'bad.png')
        with open(bad, 'wb') as f:
            f.write(b'not a png')
        args = snapshot()
        args[IDX['uov_image']] = B.file_ref(bad)
        _, missing = B.materialize(args)
        self.assertIn('unreadable', missing)

    def test_plain_jobs_pass_through(self):
        args = snapshot()
        args[IDX['uov_image']] = np.zeros((2, 2, 3), np.uint8)
        out, missing = B.materialize(args)
        self.assertEqual(missing, '')
        self.assertIs(out[IDX['uov_image']], args[IDX['uov_image']])
        self.assertFalse(B.is_file_ref({'__file__': 'x', 'extra': 1}))


class TestListHtml(unittest.TestCase):
    def test_empty_and_escaping(self):
        self.assertEqual(B.list_html([]), '')
        html = B.list_html([('/x/a<b>.png', False), ('/x/c.jpg', True)], 'the method applies')
        self.assertIn('2 image(s)', html)
        self.assertIn('a&lt;b&gt;.png', html)
        self.assertIn('the method applies', html)

    def test_long_lists_are_cut(self):
        html = B.list_html([(f'/x/{i}.png', False) for i in range(40)])
        self.assertIn('and 10 more', html)
        self.assertEqual(html.count('<li>'), 31)


class TestNaming(unittest.TestCase):
    def test_output_stem_is_file_system_safe(self):
        self.assertEqual(B.output_stem('D:/x/photo 12.jpg', 'upscaled'), 'photo 12_upscaled')
        self.assertEqual(B.output_stem('/a/b/été:ok?.PNG', 'vary'), 'été_ok_vary', 'accents kept, illegal chars replaced')
        self.assertEqual(B.output_stem('...', 'enhanced'), 'image_enhanced')
        self.assertEqual(B.output_stem('/a/b/photo.png', ''), 'photo')

    def test_unique_output_path_numbers_collisions(self):
        d = tempfile.mkdtemp(prefix='batch-out-')
        try:
            p1, n1 = B.unique_output_path(d, 'photo_upscaled', 'png')
            self.assertEqual(n1, 'photo_upscaled.png')
            open(p1, 'w').close()
            p2, n2 = B.unique_output_path(d, 'photo_upscaled', 'png')
            self.assertEqual(n2, 'photo_upscaled_2.png')
            open(p2, 'w').close()
            _, n3 = B.unique_output_path(d, 'photo_upscaled', 'png')
            self.assertEqual(n3, 'photo_upscaled_3.png')
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_suffixes(self):
        self.assertEqual(B.suffix_for('uov', 'Upscale (1.5x)'), 'upscaled')
        self.assertEqual(B.suffix_for('uov', 'Vary (Subtle)'), 'vary')
        self.assertEqual(B.suffix_for('enhance', 'Upscale (2x)'), 'enhanced')


if __name__ == '__main__':
    unittest.main()
