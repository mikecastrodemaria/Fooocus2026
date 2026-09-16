"""Guard against `import modules.x` inside a function that reads `modules.<anything>` first.

Such an import makes the name `modules` local to the whole function, so an earlier
`modules.config...` line raises UnboundLocalError at runtime (it broke every generation
once, custom-32 hotfix). flake8 does not catch it. This test walks the AST of the files
where it matters and fails on any function whose first read of `modules` comes before a
function-level `import modules.<x>` without an alias. `worker()` in async_worker.py does
its `import modules.patch` first thing, which is fine and stays allowed.

Run:  py -3.10 -m unittest tests.test_no_local_package_imports -v
"""
import ast
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES = ['modules/async_worker.py', 'webui.py']


def offending(tree, path='<src>'):
    """['file:line in fn(): import modules.x (modules read at line N)'] for each bad function."""
    found = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        imports = []
        first_read = None
        for node in ast.walk(fn):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] == 'modules' and alias.asname is None:
                        imports.append((node.lineno, alias.name))
            elif isinstance(node, ast.Name) and node.id == 'modules' and isinstance(node.ctx, ast.Load):
                if first_read is None or node.lineno < first_read:
                    first_read = node.lineno
        for lineno, name in imports:
            if first_read is not None and first_read < lineno:
                found.append(f'{path}:{lineno} in {fn.name}(): import {name} '
                             f'(modules read at line {first_read})')
    return found


def offending_in_file(path):
    with open(os.path.join(ROOT, path), encoding='utf-8') as f:
        return offending(ast.parse(f.read(), path), path)


class TestNoLocalPackageImports(unittest.TestCase):
    def test_repo_files_are_clean(self):
        found = []
        for path in FILES:
            found += offending_in_file(path)
        self.assertEqual(found, [], '\n'.join(
            ['`import modules.x` inside a function shadows the package name for the whole '
             'function; use `from modules import x` or `import modules.x as x`:'] + found))

    def test_detector_flags_a_read_before_the_import(self):
        src = ('import modules.config\n'
               'def f():\n'
               '    n = modules.config.x\n'
               '    import modules.flags\n'
               '    return n\n')
        self.assertEqual(len(offending(ast.parse(src))), 1)

    def test_detector_allows_an_import_done_first(self):
        src = ('def worker():\n'
               '    import modules.patch\n'
               '    return modules.patch.x\n')
        self.assertEqual(offending(ast.parse(src)), [])

    def test_detector_allows_aliased_and_from_imports(self):
        src = ('import modules.config\n'
               'def f():\n'
               '    n = modules.config.x\n'
               '    import modules.flags as flags\n'
               '    from modules import global_faceswap\n'
               '    return n, flags, global_faceswap\n')
        self.assertEqual(offending(ast.parse(src)), [])


if __name__ == '__main__':
    unittest.main()
