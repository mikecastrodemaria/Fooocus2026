"""custom-19 — la mise a jour au demarrage : quand elle est sure, quand elle bloque.

Travaille sur de VRAIS depots git temporaires (un depot nu joue GitHub), sans reseau.
Porte des tests crispz (tests/test_update_check.py), plus ce qui est propre a
Fooocus2026 : l'application elle-meme (avance rapide, jamais de reset) et la
question au boot.

Run:  py -3.10 -m unittest tests.test_update_check -v
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import update_check as U

ENV = {**os.environ, 'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t',
       'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@t'}


def git(cwd, *args):
    subprocess.run(['git', *args], cwd=cwd, env=ENV, check=True, capture_output=True)


def git_out(cwd, *args):
    return subprocess.run(['git', *args], cwd=cwd, env=ENV, check=True,
                          capture_output=True, text=True).stdout.strip()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


class World:
    """origin (depot nu) + work (le clone de l'utilisateur) + dev (celui qui pousse)."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix='fupd_')
        self.origin = os.path.join(self.root, 'origin.git')
        git(self.root, 'init', '--bare', '-b', 'main', self.origin)
        seed = os.path.join(self.root, 'seed')
        git(self.root, 'clone', self.origin, seed)
        git(seed, 'checkout', '-b', 'main')
        write(os.path.join(seed, 'a.txt'), 'a\n')
        write(os.path.join(seed, 'b.txt'), 'b\n')
        write(os.path.join(seed, '.gitignore'), 'outputs/\n')
        git(seed, 'add', '.')
        git(seed, 'commit', '-m', 'init')
        git(seed, 'push', '-u', 'origin', 'main')
        self.work = os.path.join(self.root, 'work')
        self.dev = os.path.join(self.root, 'dev')
        git(self.root, 'clone', self.origin, self.work)
        git(self.root, 'clone', self.origin, self.dev)
        self._old_root = U.ROOT
        U.ROOT = self.work

    def push(self, rel, text, msg, force=False):
        write(os.path.join(self.dev, rel), text)
        git(self.dev, 'add', *(['-f'] if force else []), rel)
        git(self.dev, 'commit', '-m', msg)
        git(self.dev, 'push')

    def close(self):
        U.ROOT = self._old_root
        shutil.rmtree(self.root, ignore_errors=True)


class TestAssess(unittest.TestCase):
    def setUp(self):
        self.w = World()

    def tearDown(self):
        self.w.close()

    def test_up_to_date_offers_nothing(self):
        self.assertEqual(U.assess()['status'], 'uptodate')

    def test_behind_and_clean_is_safe(self):
        self.w.push('a.txt', 'a2\n', 'change a')
        st = U.assess()
        self.assertEqual((st['status'], st['behind']), ('safe', 1), st)
        self.assertTrue(any('change a' in ln for ln in st['log']), st)
        self.assertEqual(U.main([]), 10)
        self.assertEqual(U.main(['--guard']), 0)

    def test_a_local_change_on_a_touched_file_blocks(self):
        self.w.push('a.txt', 'a2\n', 'change a')
        write(os.path.join(self.w.work, 'a.txt'), 'mon travail\n')
        st = U.assess()
        self.assertEqual((st['status'], st['overlap']), ('blocked', ['a.txt']), st)
        self.assertEqual(U.main([]), 11)
        self.assertEqual(U.main(['--guard']), 11)

    def test_a_local_change_elsewhere_is_kept(self):
        self.w.push('a.txt', 'a2\n', 'change a')
        write(os.path.join(self.w.work, 'b.txt'), 'mon travail\n')
        st = U.assess()
        self.assertEqual((st['status'], st['local']), ('safe', ['b.txt']), st)

    def test_an_ignored_file_the_update_adds_blocks(self):
        """outputs/ est ignore : git ECRASERAIT sans rien dire le fichier local."""
        self.w.push('outputs/keep.png', 'amont\n', 'add ignored', force=True)
        write(os.path.join(self.w.work, 'outputs', 'keep.png'), 'le mien\n')
        st = U.assess()
        self.assertEqual((st['status'], st['clobber']), ('blocked', ['outputs/keep.png']), st)

    def test_an_untracked_file_the_update_adds_blocks(self):
        """Le cas reel de ce depot : run_quality_rtx5090_lan.bat existe ici hors de git."""
        self.w.push('run_local.bat', 'amont\n', 'add bat')
        write(os.path.join(self.w.work, 'run_local.bat'), 'le mien\n')
        st = U.assess()
        self.assertEqual((st['status'], st['clobber']), ('blocked', ['run_local.bat']), st)

    def test_an_untracked_folder_elsewhere_does_not_block(self):
        self.w.push('a.txt', 'a2\n', 'change a')
        write(os.path.join(self.w.work, 'wildcards', '_backup', 'x.txt'), 'x\n')
        self.assertEqual(U.assess()['status'], 'safe')

    def test_a_diverged_branch_blocks(self):
        self.w.push('a.txt', 'a2\n', 'change a')
        write(os.path.join(self.w.work, 'b.txt'), 'local\n')
        git(self.w.work, 'commit', '-am', 'local commit')
        st = U.assess()
        self.assertEqual(st['status'], 'blocked', st)
        self.assertIn('divergente', st['why'])

    def test_a_local_commit_not_pushed_is_up_to_date(self):
        write(os.path.join(self.w.work, 'b.txt'), 'local\n')
        git(self.w.work, 'commit', '-am', 'local commit')
        st = U.assess()
        self.assertEqual((st['status'], st['ahead']), ('uptodate', 1), st)


class TestApply(unittest.TestCase):
    def setUp(self):
        self.w = World()

    def tearDown(self):
        self.w.close()

    def test_apply_fast_forwards_and_keeps_local_work(self):
        """L'ancien reset --hard aurait remis b.txt a 'b'."""
        self.w.push('a.txt', 'a2\n', 'change a')
        write(os.path.join(self.w.work, 'b.txt'), 'mon travail\n')
        self.assertEqual(U.assess()['status'], 'safe')
        ok, out = U.apply()
        self.assertTrue(ok, out)
        self.assertEqual(read(os.path.join(self.w.work, 'a.txt')), 'a2\n')
        self.assertEqual(read(os.path.join(self.w.work, 'b.txt')), 'mon travail\n')
        self.assertEqual(git_out(self.w.work, 'rev-parse', 'HEAD'),
                         git_out(self.w.work, 'rev-parse', '@{u}'))

    def test_boot_applies_only_when_answered_yes(self):
        self.w.push('a.txt', 'a2\n', 'change a')
        with mock.patch.object(U, 'ask', return_value=False):
            self.assertEqual(U.boot(), 'declined')
        self.assertEqual(read(os.path.join(self.w.work, 'a.txt')), 'a\n')
        with mock.patch.object(U, 'ask', return_value=True):
            self.assertEqual(U.boot(), 'updated')
        self.assertEqual(read(os.path.join(self.w.work, 'a.txt')), 'a2\n')

    def test_boot_without_console_does_not_update(self):
        self.w.push('a.txt', 'a2\n', 'change a')
        with mock.patch.object(U, 'ask', return_value=None):
            self.assertEqual(U.boot(), 'declined')
        self.assertEqual(read(os.path.join(self.w.work, 'a.txt')), 'a\n')

    def test_auto_update_applies_a_safe_update_without_asking(self):
        self.w.push('a.txt', 'a2\n', 'change a')
        with mock.patch.dict(os.environ, {'FOOOCUS_AUTO_UPDATE': '1'}), \
                mock.patch.object(U, 'ask', side_effect=AssertionError('no question')):
            self.assertEqual(U.boot(), 'updated')

    def test_auto_update_never_forces_a_blocked_update(self):
        self.w.push('a.txt', 'a2\n', 'change a')
        write(os.path.join(self.w.work, 'a.txt'), 'mon travail\n')
        with mock.patch.dict(os.environ, {'FOOOCUS_AUTO_UPDATE': '1'}):
            self.assertEqual(U.boot(), 'blocked')
        self.assertEqual(read(os.path.join(self.w.work, 'a.txt')), 'mon travail\n')

    def test_the_switch_turns_the_boot_check_off_but_not_the_guard(self):
        self.w.push('a.txt', 'a2\n', 'change a')
        write(os.path.join(self.w.work, 'a.txt'), 'mon travail\n')
        with mock.patch.dict(os.environ, {'FOOOCUS_NO_UPDATE_CHECK': '1'}):
            self.assertEqual(U.boot(), 'disabled')
            self.assertEqual(U.main([]), 0)
            self.assertEqual(U.main(['--guard']), 11)


class TestNoRepo(unittest.TestCase):
    def test_no_upstream_or_no_repo_offers_nothing(self):
        d = tempfile.mkdtemp(prefix='fupd_norepo_')
        old = U.ROOT
        try:
            U.ROOT = d
            self.assertEqual(U.assess()['status'], 'skip')
            git(d, 'init', '-b', 'main')
            write(os.path.join(d, 'a.txt'), 'a\n')
            git(d, 'add', '.')
            git(d, 'commit', '-m', 'x')
            st = U.assess()
            self.assertEqual(st['status'], 'skip', st)
            self.assertIn('branche', st['why'])
            self.assertEqual(U.main([]), 0)
        finally:
            U.ROOT = old
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
