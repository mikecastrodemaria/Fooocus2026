"""custom-21 — la file survit a un redemarrage ; un Stop ne perd plus le job.

Pur : modules.job_queue est import-safe (stdlib), numpy/PIL seulement pour les images.

Run:  py -3.10 -m unittest tests.test_job_queue_persist -v
"""
import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from modules.job_queue import JobQueue, UnserializableJob, encode_value, decode_value, QUEUE_FILE


def snapshot(prompt='a cat', n=None, image=None, inpaint=None):
    """18 ctrls factices ; n tronque pour simuler un snapshot d'une autre version."""
    args = [False, prompt, 'neg', ['Fooocus V2'], 'Speed', '1152×896', 1, 'png', 42,
            False, 2.0, 4.0, 'model.safetensors']
    args += [image, inpaint, (1, 2), np.float64(0.5), None]
    return args[:n] if n is not None else args


class TestCodec(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='fjq_')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def roundtrip(self, v):
        used = set()
        enc = encode_value(v, self.dir, used)
        return decode_value(json.loads(json.dumps(enc)), self.dir), used

    def test_images_masks_tuples_and_numpy_scalars_come_back_identical(self):
        rgb = np.random.RandomState(0).randint(0, 255, (16, 12, 3), dtype=np.uint8)
        mask = np.zeros((16, 12, 3), dtype=np.uint8)
        mask[4:8] = 255
        gray1 = rgb[:, :, :1].copy()
        floats = np.linspace(0, 1, 20, dtype=np.float32).reshape(4, 5)
        v = [rgb, {'image': rgb, 'mask': mask}, (1, 'x'), np.float64(0.25), np.bool_(True),
             gray1, floats, None, ['a', 2]]
        back, used = self.roundtrip(v)
        np.testing.assert_array_equal(back[0], rgb)
        self.assertEqual(set(back[1]), {'image', 'mask'})
        np.testing.assert_array_equal(back[1]['mask'], mask)
        self.assertEqual(back[2], (1, 'x'))
        self.assertEqual((back[3], back[4]), (0.25, True))
        self.assertEqual(back[5].shape, (16, 12, 1))
        np.testing.assert_array_equal(back[5], gray1)
        self.assertEqual(back[6].dtype, np.float32)
        np.testing.assert_array_equal(back[6], floats)
        self.assertEqual(back[7:], [None, ['a', 2]])
        self.assertEqual(len(used), 4, 'la meme image, referencee deux fois, est ecrite une fois')

    def test_an_unknown_type_is_refused(self):
        with self.assertRaises(UnserializableJob):
            encode_value(object(), self.dir, set())


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='fjq_')

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def queue(self, expected_len=None, **kw):
        q = JobQueue(max_jobs=kw.pop('max_jobs', 50))
        q.configure_persistence(self.dir, expected_len=expected_len, **kw)
        return q

    def test_the_queue_survives_a_restart_with_its_images(self):
        img = np.full((8, 8, 3), 7, dtype=np.uint8)
        q = self.queue(expected_len=18)
        q.add(snapshot('first', image=img), 'first')
        q.add(snapshot('second', inpaint={'image': img, 'mask': img}), 'second', meta={'group': 'g1', 'x': 0, 'y': 0, 'z': 0})
        q2 = self.queue(expected_len=18)
        self.assertEqual(q2.load(), 2)
        self.assertEqual(q2.labels(), ['#1 | first', '#2 | second'])
        self.assertIn('restaure', q2.status_text())
        job = q2.start_next()
        self.assertNotIn('restaure', q2.status_text(), 'le message disparait a la reprise')
        np.testing.assert_array_equal(job.args[13], img)
        self.assertEqual(q2._jobs[1].args[14]['mask'].shape, (8, 8, 3))
        self.assertEqual(q2._jobs[1].meta['group'], 'g1')

    def test_a_stopped_job_stays_first_and_a_finished_one_leaves(self):
        q = self.queue()
        q.add(snapshot('a'), 'a')
        q.add(snapshot('b'), 'b')
        job = q.start_next()
        self.assertEqual(q.labels()[0], '#1 | ▶ a')
        q.finish(job, done=False)  # Stop en plein vol
        self.assertEqual(q.labels(), ['#1 | a', '#2 | b'])
        job = q.start_next()
        q.finish(job, done=True)
        self.assertEqual(q.labels(), ['#1 | b'])
        q2 = self.queue()
        self.assertEqual(q2.load(), 1, 'ce qui est persiste suit les fins de jobs')

    def test_a_job_removed_while_running_is_not_brought_back(self):
        q = self.queue()
        q.add(snapshot('a'), 'a')
        q.add(snapshot('b'), 'b')
        job = q.start_next()
        q.remove(0)
        q.finish(job, done=False)
        self.assertEqual(q.labels(), ['#1 | b'])

    def test_orphan_images_are_cleaned_up(self):
        q = self.queue()
        q.add(snapshot('a', image=np.full((4, 4, 3), 1, dtype=np.uint8)), 'a')
        q.add(snapshot('b', image=np.full((4, 4, 3), 2, dtype=np.uint8)), 'b')
        assets = os.path.join(self.dir, 'assets')
        self.assertEqual(len(os.listdir(assets)), 2)
        q.remove(0)
        self.assertEqual(len(os.listdir(assets)), 1)
        q.clear()
        self.assertEqual(os.listdir(assets), [])

    def test_snapshots_from_another_fooocus_version_are_set_aside(self):
        q = self.queue(expected_len=18)
        q.add(snapshot('ok'), 'ok')
        q.add(snapshot('old', n=10), 'old')
        q2 = self.queue(expected_len=18)
        self.assertEqual(q2.load(), 1)
        self.assertEqual(q2.labels(), ['#1 | ok'])
        backups = [f for f in os.listdir(self.dir) if '.rejected-' in f]
        self.assertEqual(len(backups), 1)
        with open(os.path.join(self.dir, backups[0]), encoding='utf-8') as f:
            self.assertIn('au lieu de 18', json.load(f)['rejected'][0]['reason'])

    def test_a_corrupted_file_is_set_aside_not_crashing(self):
        with open(os.path.join(self.dir, QUEUE_FILE), 'w') as f:
            f.write('{not json')
        q = self.queue()
        self.assertEqual(q.load(), 0)
        self.assertTrue(any('.bad-' in f for f in os.listdir(self.dir)))

    def test_max_jobs_applies_on_restore(self):
        q = self.queue()
        for i in range(5):
            q.add(snapshot(str(i)), str(i))
        q2 = self.queue(max_jobs=3)
        self.assertEqual(q2.load(), 3)

    def test_an_unserializable_job_stays_in_memory_only(self):
        q = self.queue()
        q.add(snapshot('ok'), 'ok')
        q.add(snapshot('bad') + [object()], 'bad')
        self.assertEqual(len(q), 2)
        q2 = self.queue()
        self.assertEqual(q2.load(), 1)

    def test_xyz_groups_go_through_the_hooks(self):
        seen = {}
        q = self.queue(groups_export=lambda: {'g1': {'expected': 4}},
                       groups_import=lambda d: seen.update(d))
        q.add(snapshot('a'), 'a')
        self.assertEqual(self.queue(groups_import=lambda d: seen.update(d)).load(), 1)
        self.assertEqual(seen, {'g1': {'expected': 4}})

    def test_no_persistence_when_not_configured(self):
        q = JobQueue()
        q.add(snapshot('a'), 'a')
        self.assertFalse(q.save())
        self.assertEqual(os.listdir(self.dir), [])


class TestXyzGroups(unittest.TestCase):
    def test_groups_with_rendered_cells_survive_the_json_roundtrip(self):
        import modules.xyz_grid as xyz
        gid = 'xyz_test_roundtrip'
        group = {'id': gid, 'expected': 4, 'cells': {(0, 0, 1): 'C:/out/cell.png'},
                 'x_labels': ['a', 'b'], 'y_labels': ['1', '2'], 'z_labels': [''],
                 'nx': 2, 'ny': 2, 'nz': 1}
        xyz.register_group(group)
        try:
            data = json.loads(json.dumps(xyz.export_groups()))
            self.assertEqual(data[gid]['cells'], {'0,0,1': 'C:/out/cell.png'})
            with xyz._LOCK:
                del xyz._GROUPS[gid]
            self.assertEqual(xyz.import_groups(data), len(data))
            self.assertEqual(xyz._GROUPS[gid]['cells'], {(0, 0, 1): 'C:/out/cell.png'})
            # la case suivante complete le groupe comme si Fooocus n'avait pas redemarre
            self.assertIsNone(xyz.on_job_done({'group': gid, 'x': 0, 'y': 0, 'z': 0}, 'C:/out/c2.png'))
            self.assertEqual(len(xyz._GROUPS[gid]['cells']), 2)
        finally:
            with xyz._LOCK:
                xyz._GROUPS.pop(gid, None)


class TestSoftPause(unittest.TestCase):
    def test_pause_waits_for_the_runner_and_is_consumed_once(self):
        q = JobQueue()
        self.assertFalse(q.request_pause(), 'rien a suspendre sans runner')
        self.assertTrue(q.try_acquire_runner())
        self.assertTrue(q.request_pause())
        self.assertIn('Pause demandee', q.status_text())
        self.assertTrue(q.consume_pause_request())
        self.assertTrue(q.paused)
        self.assertFalse(q.consume_pause_request())


if __name__ == '__main__':
    unittest.main()
