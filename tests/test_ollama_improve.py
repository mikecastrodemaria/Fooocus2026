"""custom-28 — Improve prompt via Ollama, contre un faux serveur Ollama local (stdlib).

Run:  py -3.10 -m unittest tests.test_ollama_improve -v
"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from modules import ollama_improve as I


class FakeOllama(BaseHTTPRequestHandler):
    last_generate = None
    reply = '<think>let me polish</think>"a lone red fox, snowy pine forest, soft dawn light"'

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
            self._send({'models': [{'name': 'qwen3:8b'}, {'name': 'llama3.1:8b'}]})
        else:
            self._send({'error': 'nope'}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        if self.path == '/api/generate':
            if body.get('model') == 'missing:1b':
                return self._send({'error': 'model not found'}, 404)
            if 'think' in body:   # un modele sans raisonnement refuse `think` (400) -> rejeu
                return self._send({'error': 'model does not support thinking'}, 400)
            FakeOllama.last_generate = body
            return self._send({'response': FakeOllama.reply})
        self._send({'error': 'nope'}, 404)


class TestImprove(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(('127.0.0.1', 0), FakeOllama)
        cls.base = f'http://127.0.0.1:{cls.httpd.server_address[1]}'
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def test_positive_rewrite_is_stripped_of_thinking_and_quotes(self):
        out, model = I.improve('a fox', kind='positive', model='llama3.1:8b', base=self.base)
        self.assertEqual(model, 'llama3.1:8b')
        self.assertEqual(out, 'a lone red fox, snowy pine forest, soft dawn light')
        body = FakeOllama.last_generate
        self.assertIn('PROMPT: a fox', body['prompt'])
        self.assertNotIn('think', body, 'rejoue sans think apres le 400')

    def test_negative_uses_the_negative_instruction(self):
        I.improve('blurry', kind='negative', model='llama3.1:8b', base=self.base)
        self.assertIn('NEGATIVE PROMPT: blurry', FakeOllama.last_generate['prompt'])

    def test_no_model_given_picks_the_first_installed(self):
        _, model = I.improve('a fox', base=self.base)
        self.assertEqual(model, 'qwen3:8b')

    def test_list_models_returns_every_installed_model(self):
        self.assertEqual(I.list_models(base=self.base), ['qwen3:8b', 'llama3.1:8b'])

    def test_empty_prompt_is_a_clean_error_not_a_call(self):
        with self.assertRaises(I.OllamaError):
            I.improve('   ', base=self.base)

    def test_a_missing_model_says_how_to_get_it(self):
        with self.assertRaises(I.OllamaError) as cm:
            I.improve('a fox', model='missing:1b', base=self.base)
        self.assertIn('ollama pull missing:1b', str(cm.exception))

    def test_ollama_down_gives_an_actionable_message(self):
        with self.assertRaises(I.OllamaError) as cm:
            I.improve('a fox', model='llama3.1:8b', base='http://127.0.0.1:9', timeout=2)
        self.assertIn('Ollama unreachable', str(cm.exception))

    def test_a_reply_that_is_only_thinking_is_an_error(self):
        old = FakeOllama.reply
        FakeOllama.reply = '<think>endless reasoning'
        try:
            with self.assertRaises(I.OllamaError):
                I.improve('a fox', model='llama3.1:8b', base=self.base)
        finally:
            FakeOllama.reply = old


class TestInstructions(unittest.TestCase):
    def test_positive_and_negative_are_distinct(self):
        self.assertIn('PROMPT:', I._instruction('positive'))
        self.assertNotIn('NEGATIVE', I._instruction('positive'))
        self.assertIn('NEGATIVE PROMPT:', I._instruction('negative'))


if __name__ == '__main__':
    unittest.main()
