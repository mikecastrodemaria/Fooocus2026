"""custom-25 — Describe via Ollama, contre un faux serveur Ollama local (stdlib).

Run:  py -3.10 -m unittest tests.test_ollama_describe -v
"""
import base64
import io
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from modules import ollama_describe as D


class FakeOllama(BaseHTTPRequestHandler):
    last_generate = None
    reply = '<think>let me look</think>A pencil portrait of a man. No text is visible. It appears to be night.'

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == '/api/tags':
            self._send({'models': [{'name': 'qwen3:8b'}, {'name': 'mystery-vl:4b'}, {'name': 'llava:7b'}]})
        else:
            self._send({'error': 'nope'}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        if self.path == '/api/show':
            caps = {'qwen3:8b': ['completion'], 'mystery-vl:4b': ['completion', 'vision'],
                    'llava:7b': ['completion', 'vision']}.get(body.get('model'), [])
            return self._send({'capabilities': caps})
        if self.path == '/api/generate':
            if body.get('model') == 'missing:1b':
                return self._send({'error': 'model not found'}, 404)
            if 'think' in body:
                return self._send({'error': 'model does not support thinking'}, 400)
            FakeOllama.last_generate = body
            return self._send({'response': FakeOllama.reply})
        self._send({'error': 'nope'}, 404)


class TestDescribe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(('127.0.0.1', 0), FakeOllama)
        cls.base = f'http://127.0.0.1:{cls.httpd.server_address[1]}'
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def image(self):
        return np.full((1600, 900, 3), 128, dtype=np.uint8)

    def test_a_prose_description_is_cleaned_and_the_image_is_bounded(self):
        text, model = D.describe(self.image(), model='llava:7b', style='Prompt (prose)',
                                 length='Long', base=self.base)
        self.assertEqual(model, 'llava:7b')
        self.assertEqual(text, 'A pencil portrait of a man. It is night.')
        body = FakeOllama.last_generate
        self.assertIn('about 180 words', body['prompt'])
        self.assertIn('Begin with the medium and style', body['prompt'])
        self.assertNotIn('think', body, 'rejoue sans think apres le 400')
        img = Image.open(io.BytesIO(base64.b64decode(body['images'][0])))
        self.assertEqual(max(img.size), 1024)

    def test_vision_models_come_from_capabilities_known_names_first(self):
        self.assertEqual(D.list_vision_models(base=self.base), ['mystery-vl:4b', 'llava:7b'])

    def test_no_model_given_picks_the_first_vision_model(self):
        _, model = D.describe(self.image(), base=self.base)
        self.assertEqual(model, 'mystery-vl:4b')

    def test_a_missing_model_says_how_to_get_it(self):
        with self.assertRaises(D.OllamaError) as cm:
            D.describe(self.image(), model='missing:1b', base=self.base)
        self.assertIn('ollama pull missing:1b', str(cm.exception))

    def test_ollama_down_gives_an_actionable_message(self):
        with self.assertRaises(D.OllamaError) as cm:
            D.describe(self.image(), model='llava:7b', base='http://127.0.0.1:9', timeout=2)
        self.assertIn('Ollama unreachable', str(cm.exception))

    def test_default_endpoint_is_ipv4_loopback(self):
        # custom-36 : "localhost" resolves to ::1 first for Python on Windows while Ollama
        # listens on IPv4 only; the attempt hangs until the timeout. 127.0.0.1 never does.
        self.assertEqual(D.DEFAULT_ENDPOINT, 'http://127.0.0.1:11434')

    def test_a_proxy_in_the_environment_is_ignored(self):
        # custom-35 : HTTP_PROXY / HTTPS_PROXY (Pinokio, corporate networks) must never
        # capture the call to a local Ollama; with the default urllib opener it did.
        import os
        saved = {k: os.environ.get(k) for k in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy',
                                                 'https_proxy', 'NO_PROXY', 'no_proxy')}
        try:
            for k in saved:
                os.environ.pop(k, None)
            os.environ['HTTP_PROXY'] = 'http://127.0.0.1:9'
            os.environ['http_proxy'] = 'http://127.0.0.1:9'
            tags = D._http('/api/tags', base=self.base, timeout=5)
            self.assertIn('models', tags)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_a_reply_that_is_only_thinking_is_an_error_not_an_empty_prompt(self):
        old = FakeOllama.reply
        FakeOllama.reply = '<think>endless reasoning'
        try:
            with self.assertRaises(D.OllamaError):
                D.describe(self.image(), model='llava:7b', base=self.base)
        finally:
            FakeOllama.reply = old


class TestInstructions(unittest.TestCase):
    def test_styles_and_lengths(self):
        self.assertIn('about 60 words', D.describe_instruction('Photo (technical)', 'Short'))
        self.assertIn('at most 25 words', D.describe_instruction(D.SHORT_CAPTION_STYLE, 'Very long'))
        self.assertEqual(D.describe_instruction('unknown', 'bogus'), D.describe_instruction())
        for style in D.DESCRIBE_STYLES:
            self.assertNotIn('{words}', D.describe_instruction(style))

    def test_cleaning_keeps_a_tag_list_and_strips_orphan_thinking(self):
        self.assertEqual(D.clean_description('ink, no text, girl'), 'ink, no text, girl')
        self.assertEqual(D.strip_thinking('reasoning here</think> A cat.'), 'A cat.')
        self.assertEqual(D.clean_description('A dog, probably a beagle.'), 'A dog a beagle.')


if __name__ == '__main__':
    unittest.main()
