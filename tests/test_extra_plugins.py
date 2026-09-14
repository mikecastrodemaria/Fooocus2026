"""custom-20 — plugins Extra : mode serveur et mise a jour.

Aucun GPU, aucun reseau :
  - le serveur est un faux serveur HTTP (stdlib) qui imite l'API de la famille
    crispz (/health, /upscale, /unload) ;
  - la mise a jour travaille sur de vrais depots git, clone superficiel comme a
    l'install (--depth 1), avec un depot nu qui joue GitHub.

Run:  py -3.10 -m unittest tests.test_extra_plugins -v
"""
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

from extra_plugins import installer, runner, server

ENV = {**os.environ, 'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t',
       'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@t'}

STUB = r'''
import json, os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
argv = sys.argv
if "--fail" in argv:
    print("boom: fastapi missing", flush=True)
    sys.exit(3)
port = int(argv[argv.index("--port") + 1])
esrgan = argv[argv.index("--esrgan-dir") + 1] if "--esrgan-dir" in argv else ""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok", "esrgan": esrgan, "pid": os.getpid()})
        else:
            self._send({"detail": "nope"}, 404)
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/upscale":
            if not os.path.isfile(body.get("input", "")):
                return self._send({"detail": "input not found"}, 400)
            out = os.path.join(body["output_dir"], "out.json")
            with open(out, "w") as f:
                json.dump(body, f)
            self._send({"output": out, "size": [2, 2], "esrgan_s": 0.1, "refine_s": 0.2,
                        "total_s": 0.3})
        elif self.path == "/unload":
            with open("unloads.txt", "a") as f:
                f.write("x")
            self._send({"status": "unloaded"})
        else:
            self._send({"detail": "nope"}, 404)

HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def plugin_manifest(port=None, launch_extra=()):
    m = {
        'manifest_version': 1, 'id': 'demo', 'name': 'Demo', 'version': '1',
        'entry': {
            'command': ['{venv_python}', 'app.py', '--cli'], 'input_arg': '-i',
            'output': {'mode': 'print_output', 'flag': '--print-output',
                       'save_mode_flag': ['--save-mode', 'custom'],
                       'save_dir_flag': ['--output-dir', '{output_dir}'],
                       'format_flag': ['--output-format', 'png']},
            'models_arg': ['--esrgan-dir', '{esrgan_dir}'],
        },
        'params': [
            {'key': 'model', 'type': 'dropdown', 'arg': '-m', 'choices_cmd': ['--list-models']},
            {'key': 'factor', 'type': 'slider', 'arg': '--factor'},
            {'key': 'seed', 'type': 'number', 'arg': '--seed'},
            {'key': 'prompt', 'type': 'text', 'arg': '--prompt'},
        ],
        'env': {'strategies': {'fresh_venv': {'create': [], 'steps': [
            {'name': 'torch', 'cmd': [sys.executable, '-c', 'pass']},
            {'name': 'deps', 'cmd': [sys.executable, '-c',
                                     "open('deps_ran.txt','a').write('x')",
                                     '-r', 'requirements.txt']},
        ]}}},
    }
    if port is not None:
        m['server'] = {'launch': [sys.executable, 'stub.py', '--host', '127.0.0.1',
                                  '--port', str(port), *launch_extra]}
    return m


class TestServerPayload(unittest.TestCase):
    def test_payload_uses_keys_native_types_and_cli_output_mode(self):
        m = plugin_manifest()
        p = runner.build_server_payload(m, 'C:/in.png', 'C:/out', {
            'model': '', 'factor': 2.0, 'seed': -1.0, 'prompt': '', 'unknown': 1})
        self.assertEqual(p, {'input': 'C:/in.png', 'save_mode': 'custom', 'output_dir': 'C:/out',
                             'output_format': 'png', 'factor': 2, 'seed': -1, 'prompt': ''})
        self.assertIsInstance(p['factor'], int)

    def test_url_from_launch_flags(self):
        m = plugin_manifest(port=7861)
        self.assertEqual(server.server_url(m), 'http://127.0.0.1:7861')
        self.assertIsNone(server.server_url(plugin_manifest()))


class TestServerLifecycle(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix='fsrv_')
        with open(os.path.join(self.dir, 'stub.py'), 'w') as f:
            f.write(STUB)
        self.inp = os.path.join(self.dir, 'in.png')
        with open(self.inp, 'wb') as f:
            f.write(b'png')
        self.port = free_port()

    def tearDown(self):
        server.stop_all()
        shutil.rmtree(self.dir, ignore_errors=True)

    def plugin(self, *launch_extra):
        return {'id': 'demo', 'dir': self.dir,
                'manifest': plugin_manifest(self.port, launch_extra)}

    def test_start_upscale_warm_release_and_stop(self):
        p = self.plugin()
        url = server.ensure(p, start_timeout=30)
        self.assertTrue(server.is_running('demo'))
        payload = runner.build_server_payload(p['manifest'], self.inp, self.dir, {'factor': 2.0})
        r1 = server.upscale('demo', url, payload)
        self.assertFalse(r1['was_warm'])
        with open(r1['output']) as f:
            self.assertEqual(json.load(f)['factor'], 2)
        r2 = server.upscale('demo', server.ensure(p, start_timeout=30), payload)
        self.assertTrue(r2['was_warm'], 'le 2e appel reutilise le meme serveur')
        self.assertEqual(server.release_vram_all(), ['demo'])
        self.assertEqual(server.release_vram_all(), [], 'rien a rendre deux fois')
        with open(os.path.join(self.dir, 'unloads.txt')) as f:
            self.assertEqual(f.read(), 'x')
        proc = server._SERVERS['demo'].proc
        self.assertTrue(server.stop('demo'))
        self.assertIsNotNone(proc.poll())

    def test_a_server_error_is_reported_with_detail(self):
        p = self.plugin()
        url = server.ensure(p, start_timeout=30)
        payload = runner.build_server_payload(p['manifest'], 'C:/nope.png', self.dir, {})
        with self.assertRaises(server.ServerError) as cm:
            server.upscale('demo', url, payload)
        self.assertIn('input not found', str(cm.exception))

    def test_a_server_that_dies_at_start_reports_its_log(self):
        with self.assertRaises(server.ServerError) as cm:
            server.ensure(self.plugin('--fail'), start_timeout=30)
        self.assertIn('fastapi missing', str(cm.exception))
        self.assertFalse(server.is_running('demo'))

    def test_changing_the_esrgan_folder_restarts_the_server(self):
        p = self.plugin()
        server.ensure(p, esrgan_dir='A', start_timeout=30)
        first = server._SERVERS['demo'].proc
        url = server.ensure(p, esrgan_dir='B', start_timeout=30)
        self.assertIsNotNone(first.poll(), "l'ancien serveur est arrete")
        self.assertEqual(server.health(url)['esrgan'], 'B')

    def test_an_external_server_is_reused_and_never_killed(self):
        ext = subprocess.Popen([sys.executable, 'stub.py', '--port', str(self.port)], cwd=self.dir)
        try:
            url = server.server_url(self.plugin()['manifest'])
            for _ in range(60):
                if server.health(url):
                    break
                time.sleep(0.25)
            self.assertEqual(server.ensure(self.plugin(), start_timeout=30), url)
            self.assertFalse(server._SERVERS['demo'].owned)
            server.stop('demo')
            self.assertIsNone(ext.poll(), 'le serveur externe tourne toujours')
        finally:
            ext.terminate()
            ext.wait(10)


def git(cwd, *args):
    subprocess.run(['git', *args], cwd=cwd, env=ENV, check=True, capture_output=True)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


class PluginWorld:
    """origin (depot nu du plugin) + dev (qui pousse) + plugin (clone --depth 1 installe)."""

    def __init__(self):
        self.root = tempfile.mkdtemp(prefix='fplug_')
        self.origin = os.path.join(self.root, 'origin.git')
        git(self.root, 'init', '--bare', '-b', 'main', self.origin)
        self.dev = os.path.join(self.root, 'dev')
        git(self.root, 'clone', self.origin, self.dev)
        git(self.dev, 'checkout', '-b', 'main')
        write(os.path.join(self.dev, 'fooocus_extra.json'), json.dumps(plugin_manifest(), indent=1))
        write(os.path.join(self.dev, 'app.py'), 'v1\n')
        write(os.path.join(self.dev, 'requirements.txt'), 'pkg==1\n')
        write(os.path.join(self.dev, '.gitignore'), 'deps_ran.txt\n')
        git(self.dev, 'add', '.')
        git(self.dev, 'commit', '-m', 'init')
        git(self.dev, 'push', '-u', 'origin', 'main')
        self.plugin = os.path.join(self.root, 'plugin')
        git(self.root, 'clone', '--depth', '1', pathlib.Path(self.origin).as_uri(), self.plugin)

    def push(self, rel, text, msg):
        write(os.path.join(self.dev, rel), text)
        git(self.dev, 'add', rel)
        git(self.dev, 'commit', '-m', msg)
        git(self.dev, 'push')

    def deps_ran(self):
        return os.path.isfile(os.path.join(self.plugin, 'deps_ran.txt'))

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)


class TestPluginUpdate(unittest.TestCase):
    def setUp(self):
        self.w = PluginWorld()
        self.log = []

    def tearDown(self):
        self.w.close()

    def update(self, **kw):
        return installer.update_plugin(self.w.plugin, log=self.log.append, **kw)

    def test_up_to_date_does_nothing(self):
        self.assertEqual(installer.update_status(self.w.plugin)['status'], 'uptodate')
        res = self.update()
        self.assertEqual((res['updated'], res['deps']), (False, False))
        self.assertFalse(self.w.deps_ran())

    def test_a_code_change_updates_without_reinstalling_deps(self):
        self.w.push('app.py', 'v2\n', 'fix blackwell')
        st = installer.update_status(self.w.plugin)
        self.assertEqual((st['status'], st['behind']), ('safe', 1), st)
        res = self.update()
        self.assertEqual((res['updated'], res['deps'], res['manifest_changed']), (True, False, False))
        with open(os.path.join(self.w.plugin, 'app.py')) as f:
            self.assertEqual(f.read(), 'v2\n')
        self.assertFalse(self.w.deps_ran())

    def test_a_requirements_change_reinstalls_deps_only(self):
        self.w.push('requirements.txt', 'pkg==2\n', 'bump pkg')
        res = self.update()
        self.assertTrue(res['deps'])
        with open(os.path.join(self.w.plugin, 'deps_ran.txt')) as f:
            self.assertEqual(f.read(), 'x', "l'etape torch n'est jamais rejouee, deps une fois")

    def test_force_deps_reinstalls_even_when_up_to_date(self):
        res = self.update(force_deps=True)
        self.assertEqual((res['updated'], res['deps']), (False, True))

    def test_a_local_change_on_a_touched_file_blocks_and_touches_nothing(self):
        self.w.push('app.py', 'v2\n', 'change app')
        write(os.path.join(self.w.plugin, 'app.py'), 'mon patch\n')
        self.assertEqual(installer.update_status(self.w.plugin)['status'], 'blocked')
        with self.assertRaises(RuntimeError):
            self.update()
        with open(os.path.join(self.w.plugin, 'app.py')) as f:
            self.assertEqual(f.read(), 'mon patch\n')

    def test_a_manifest_change_is_reported(self):
        m = plugin_manifest()
        m['version'] = '2'
        self.w.push('fooocus_extra.json', json.dumps(m, indent=1), 'manifest v2')
        self.assertTrue(self.update()['manifest_changed'])

    def test_a_non_git_plugin_is_skipped(self):
        d = tempfile.mkdtemp(prefix='fplug_nogit_')
        try:
            self.assertEqual(installer.update_status(d)['status'], 'skip')
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
