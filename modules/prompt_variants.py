"""custom-29 — {a|b|c} variant groups in prompts (dynamic-prompts syntax).

Expands the brace syntax popularised by the Automatic1111 "dynamic prompts" extension,
next to Fooocus's own `__wildcard__` placeholders:

    {a|b|c}            one option, chosen at random (seed-bound) or in order
    {a|}               an empty option: 50 % chance of nothing
    {2$$a|b|c}         two distinct options, joined with ", "
    {1-3$$a|b|c}       between one and three options
    {2$$ and $$a|b|c}  custom separator between the picked options
    {a|{b|c}}          groups nest; innermost first

A group is only expanded when it contains a `|` or a `$$` count prefix, so `{prompt}`
(the style placeholder) and stray braces stay untouched. Standard library only, no
Fooocus import: callable from modules.util.apply_wildcards and testable on its own.
"""
import re

# Innermost group only (no brace inside), so nested groups resolve inside-out.
_GROUP_RE = re.compile(r'\{([^{}]*)\}')
# "2$$" or "1-3$$" at the start of a group body.
_COUNT_RE = re.compile(r'^\s*(\d+)(?:\s*-\s*(\d+))?\s*\$\$')
# Anything the dynamic syntax could act on: a variant group or a wildcard placeholder.
_SYNTAX_RE = re.compile(r'\{[^{}]*(?:\||\$\$)[^{}]*\}|__[\w-]+__')


def has_variants(text):
    """True when `text` still holds at least one expandable {…|…} or {N$$…} group."""
    return any(_is_group(m.group(1)) for m in _GROUP_RE.finditer(text or ''))


def uses_dynamic_syntax(text):
    """True when `text` holds a variant group or a __wildcard__ placeholder (for Improve)."""
    return bool(_SYNTAX_RE.search(text or ''))


def _is_group(body):
    return '|' in body or _COUNT_RE.match(body) is not None


def _expand_group(match, rng, index, in_order):
    body = match.group(1)
    if not _is_group(body):
        return match.group(0)                       # {prompt} and friends: leave as is

    lo = hi = 1
    sep = ', '
    count = _COUNT_RE.match(body)
    if count:
        lo = int(count.group(1))
        hi = int(count.group(2)) if count.group(2) else lo
        body = body[count.end():]
        if '$$' in body:                            # {2$$ and $$a|b}: custom separator
            sep, body = body.split('$$', 1)

    options = [o.strip() for o in body.split('|')]
    n = len(options)

    if not count:
        return options[index % n] if in_order else rng.choice(options)

    lo = max(0, min(lo, n))
    hi = max(lo, min(hi, n))
    if lo == hi:
        k = lo
    elif in_order:
        k = lo + index % (hi - lo + 1)
    else:
        k = rng.randint(lo, hi)
    if in_order:
        picks = [options[(index + j) % n] for j in range(k)]
    else:
        picks = rng.sample(options, k)
    return sep.join(p for p in picks if p)


def expand_variants(text, rng, index=0, in_order=False, max_depth=64):
    """Expand every variant group in `text`, innermost first.

    `rng` is the per-image random.Random (seed-bound: same seed, same picks).
    `index` is the image number in the batch; with `in_order` the options are walked
    top to bottom like `Read wildcards in order` does for wildcard files.
    """
    for _ in range(max_depth):
        expanded = _GROUP_RE.sub(lambda m: _expand_group(m, rng, index, in_order), text)
        if expanded == text:
            return text
        text = expanded
    return text
