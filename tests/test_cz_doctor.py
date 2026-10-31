"""crispz-sdxl: czp doctor against fake instances (down, legacy custom-26,
crispz-sdxl, another crispz app), on CPU.

Run:  python -m pytest tests/test_cz_doctor.py -q
"""
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

sys.argv = sys.argv[:1]

import cz_doctor  # noqa: E402
import fooocus_protocol as P  # noqa: E402


class Instance(BaseHTTPRequestHandler):
    caps = None  # dict answered by cli_caps, or None for a 404

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers.get('Content-Length') or 0))
        if self.path != '/run/cli_caps' or Instance.caps is None:
            self.send_response(404)
            self.end_headers()
            return
        b = json.dumps({'data': [json.dumps(Instance.caps)]}).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class TestDoctor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(('127.0.0.1', 0), Instance)
        cls.url = f'http://127.0.0.1:{cls.httpd.server_address[1]}'
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.tmp = tempfile.mkdtemp(prefix='czdoc_')

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def run_doctor(self, url):
        buf = io.StringIO()
        with mock.patch.dict(os.environ, {'CRISPZ_OUT_DIR': self.tmp, 'CRISPZ_FAKE_ENGINE': '0'}):
            code = P.main(['doctor', '--remote', url], out=buf)
        return code, buf.getvalue()

    def test_nothing_listening_says_to_start_the_app(self):
        code, text = self.run_doctor('http://127.0.0.1:9')
        self.assertEqual(code, 1)
        self.assertIn('nothing answers', text)
        self.assertIn('Start run.bat', text)
        self.assertIn(cz_doctor.BOOT_LINE, text)

    def test_legacy_fooocus_says_to_restart_on_the_branch(self):
        Instance.caps = {'ok': True, 'tool': 'fooocus2026', 'version': '2026.7.0'}
        code, text = self.run_doctor(self.url)
        self.assertEqual(code, 1)
        self.assertIn('custom-26 endpoints only', text)
        self.assertIn('answers as fooocus2026', text)

    def test_another_crispz_app_is_named(self):
        Instance.caps = {'ok': True, 'tool': 'crispz-studio', 'version': '1.0'}
        code, text = self.run_doctor(self.url)
        self.assertEqual(code, 1)
        self.assertIn('crispz-studio', text)
        self.assertIn('another crispz app', text)

    def test_no_endpoint_is_told_apart_from_down(self):
        Instance.caps = None
        code, text = self.run_doctor(self.url)
        self.assertEqual(code, 1)
        self.assertIn('no cli_caps endpoint', text)

    def test_a_crispz_sdxl_instance_passes_when_the_rest_is_fine(self):
        Instance.caps = {'ok': True, 'tool': 'crispz-sdxl', 'version': '0.1.0',
                         'engine': {'tool': 'fooocus2026', 'version': '2026.7.0', 'fake': False},
                         'ops': ['caps', 'gen', 'edit', 'faces'], 'models': ['a.safetensors'], 'loras': [],
                         'hidden': {'checkpoints': {'sd': 3}}, 'model_loaded': True, 'model_switch': False,
                         'supports': {'faces': True}}
        with mock.patch.object(cz_doctor, '_git', side_effect=lambda *a: 'wip/crispz-sdxl' if 'HEAD' in a and
                               '--abbrev-ref' in a else 'abc1234'), \
                mock.patch.object(cz_doctor, '_module_version', return_value='1.0'):
            code, text = self.run_doctor(self.url)
        self.assertEqual(code, 0, text)
        self.assertIn('crispz-sdxl 0.1.0 on fooocus2026 2026.7.0', text)
        self.assertIn('1 SDXL checkpoints', text)
        self.assertIn('All good', text)

    def test_wrong_branch_is_a_problem(self):
        Instance.caps = {'ok': True, 'tool': 'crispz-sdxl', 'version': '0.1.0', 'engine': {},
                         'supports': {'faces': True}}
        with mock.patch.object(cz_doctor, '_git', return_value='main'), \
                mock.patch.object(cz_doctor, '_module_version', return_value='1.0'):
            code, text = self.run_doctor(self.url)
        self.assertEqual(code, 1)
        self.assertIn('git checkout wip/crispz-sdxl', text)


if __name__ == '__main__':
    unittest.main()
