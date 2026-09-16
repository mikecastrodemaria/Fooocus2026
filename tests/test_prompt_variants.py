"""custom-29 — {a|b|c} variant groups (dynamic-prompts syntax), standard library only.

Run:  py -3.10 -m unittest tests.test_prompt_variants -v
"""
import random
import unittest

from modules import prompt_variants as V


def expand(text, seed=1, index=0, in_order=False):
    return V.expand_variants(text, random.Random(seed), index=index, in_order=in_order)


class TestSingleChoice(unittest.TestCase):
    def test_picks_one_of_the_options(self):
        for seed in range(20):
            self.assertIn(expand('a {shy|sad|smile} girl', seed=seed),
                          ('a shy girl', 'a sad girl', 'a smile girl'))

    def test_same_seed_same_pick(self):
        self.assertEqual(expand('{a|b|c|d|e|f}', seed=7), expand('{a|b|c|d|e|f}', seed=7))

    def test_different_seeds_eventually_differ(self):
        picks = {expand('{a|b|c|d|e|f}', seed=s) for s in range(30)}
        self.assertGreater(len(picks), 1)

    def test_in_order_walks_the_options_by_image_index(self):
        out = [expand('{shy|sad|smile}', index=i, in_order=True) for i in range(5)]
        self.assertEqual(out, ['shy', 'sad', 'smile', 'shy', 'sad'])

    def test_whitespace_around_options_is_trimmed(self):
        self.assertEqual(expand('{ shy | sad }', index=1, in_order=True), 'sad')

    def test_empty_option_can_yield_nothing(self):
        self.assertEqual(expand('a {smiling|} face', index=1, in_order=True), 'a  face')
        self.assertEqual(expand('a {smiling|} face', index=0, in_order=True), 'a smiling face')

    def test_several_groups_in_one_prompt(self):
        out = expand('{red|blue} car, {day|night}', index=1, in_order=True)
        self.assertEqual(out, 'blue car, night')


class TestNesting(unittest.TestCase):
    def test_inner_group_resolves_first(self):
        out = expand('{a|{b|c}}', index=1, in_order=True)
        self.assertEqual(out, 'c')  # inner picks 'c' (index 1), outer picks option 1 = that

    def test_deep_nesting_terminates(self):
        text = '{x|' * 10 + 'y' + '}' * 10
        out = expand(text)
        self.assertIn(out, ('x', 'y'))
        self.assertNotIn('{', out)

    def test_depth_limit_returns_partial_text_instead_of_hanging(self):
        text = '{x|' * 5 + 'y' + '}' * 5
        out = V.expand_variants(text, random.Random(1), max_depth=1)
        self.assertIn('{', out)


class TestMultiPick(unittest.TestCase):
    def test_fixed_count_picks_distinct_options(self):
        out = expand('{2$$shy|sad|smile}', seed=3)
        parts = out.split(', ')
        self.assertEqual(len(parts), 2)
        self.assertEqual(len(set(parts)), 2)
        self.assertTrue(set(parts) <= {'shy', 'sad', 'smile'})

    def test_range_count_stays_inside_the_range(self):
        for seed in range(30):
            n = len(expand('{1-2$$a|b|c}', seed=seed).split(', '))
            self.assertIn(n, (1, 2))

    def test_custom_separator(self):
        out = expand('{2$$ and $$a|b|c}', index=0, in_order=True)
        self.assertEqual(out, 'a and b')

    def test_count_capped_to_the_number_of_options(self):
        self.assertEqual(sorted(expand('{5$$a|b}', seed=1).split(', ')), ['a', 'b'])

    def test_in_order_multi_pick_walks_from_the_image_index(self):
        self.assertEqual(expand('{2$$a|b|c}', index=2, in_order=True), 'c, a')

    def test_zero_count_gives_nothing(self):
        self.assertEqual(expand('x{0$$a|b}y', seed=1), 'xy')


class TestLeftAlone(unittest.TestCase):
    def test_style_placeholder_is_untouched(self):
        self.assertEqual(expand('{prompt}, cinematic'), '{prompt}, cinematic')

    def test_unclosed_group_is_untouched(self):
        text = '{disdainful|disgust|shy|sad'
        self.assertEqual(expand(text), text)

    def test_wildcard_placeholder_is_untouched(self):
        self.assertEqual(expand('__color__ {cat|dog}', index=0, in_order=True), '__color__ cat')

    def test_plain_text_makes_no_rng_call(self):
        rng = random.Random(1)
        before = rng.getstate()
        V.expand_variants('a plain prompt', rng)
        self.assertEqual(rng.getstate(), before)

    def test_none_and_empty(self):
        self.assertEqual(V.expand_variants('', random.Random(1)), '')
        self.assertFalse(V.has_variants(None))


class TestDetection(unittest.TestCase):
    def test_has_variants(self):
        self.assertTrue(V.has_variants('{a|b}'))
        self.assertTrue(V.has_variants('{2$$a|b}'))
        self.assertFalse(V.has_variants('{prompt}'))
        self.assertFalse(V.has_variants('plain'))

    def test_uses_dynamic_syntax_includes_wildcards(self):
        self.assertTrue(V.uses_dynamic_syntax('__color__ flower'))
        self.assertTrue(V.uses_dynamic_syntax('{a|b}'))
        self.assertFalse(V.uses_dynamic_syntax('{prompt} only'))
        self.assertFalse(V.uses_dynamic_syntax('a fox'))


if __name__ == '__main__':
    unittest.main()
