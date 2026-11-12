"""R006 unit test for search_r1.plan.lexical_phi.

Run: python scripts/plan/r006_lexical_test.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from search_r1.plan.lexical_phi import extract_docs, lexical_phi, lexical_phi_from_state

DOC = 'Doc 1(Title: "The Beatles") The Beatles were an English rock band formed in Liverpool in 1960.\n'

CASES = [
    # (name, docs, golden_answers, expected)
    ('exact alias', [DOC], ['Liverpool'], 1),
    ('case and punctuation', [DOC], ['LIVERPOOL!'], 1),
    ('article stripped', [DOC], ['Beatles'], 1),
    ('article in alias', ['beatles rock band'], ['The Beatles'], 1),
    ('multi-word alias', [DOC], ['English rock band'], 1),
    ('any alias of several', [DOC], ['Manchester', 'Liverpool'], 1),
    ('alias as str', [DOC], 'Liverpool', 1),
    ('absent answer', [DOC], ['Manchester'], 0),
    ('numeric whole token', [DOC], ['1960'], 1),
    ('numeric no partial: 1 vs 1960', [DOC], ['1'], 0),
    ('word no partial: he vs the', ['the band'], ['he'], 0),
    ('word no partial: pool vs Liverpool', [DOC], ['pool'], 0),
    ('alias empty after normalisation', [DOC], ['The', '!!'], 0),
    ('empty alias list', [DOC], [], 0),
    ('no docs', [], ['Liverpool'], 0),
    ('second doc hits', ['nothing here', 'born in Liverpool.'], ['Liverpool'], 1),
]


def main():
    fails = 0
    for name, docs, gold, exp in CASES:
        got = lexical_phi(docs, gold)
        ok = got == exp
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: got {got}, expected {exp}")

    # cumulative docs from a multi-turn state; question and model text are not matched
    state = ('Question: where was the band formed? Liverpool? <search>beatles</search>\n'
             '<information>Doc 1(Title: X) unrelated text</information>\n'
             '<think>maybe Liverpool</think><search>beatles origin</search>\n'
             f'<information>{DOC}</information>\n')
    checks = [
        ('extract_docs finds both blocks', len(extract_docs(state)) == 2),
        ('cumulative: hit in later block', lexical_phi_from_state(state, ['Liverpool']) == 1),
        ('only first block: no hit', lexical_phi(extract_docs(state)[:1], ['Liverpool']) == 0),
        ('question/think text ignored', lexical_phi_from_state(
            state.replace(DOC, 'nothing'), ['Liverpool']) == 0),
    ]
    for name, ok in checks:
        fails += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")

    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
