"""CPU test of the reader inputs and scorer (checklist item 6).

  1. per-doc truncation to 256 tokens, docs block <= 2048 tokens, earliest docs dropped first;
  2. identical passages kept once; "Document n:" numbering follows the kept docs;
  3. build(): one row per distinct alias, context identical across aliases, target = alias tokens;
  4. collate(): target mask covers exactly the alias tokens; right padding;
  5. PhiScorer (tiny random Qwen2, CPU) equals the mean target log-prob from full logits,
     and is unchanged by extra right padding;
  6. negctrl: answer classes, the map is a within-class derangement, deterministic for seed 0.

Run: HF_HUB_OFFLINE=1 python -I scripts/plan/item6_reader_test.py
(the 3B tokenizer has the same vocabulary as Qwen2.5-7B-Instruct; override with PLAN_TEST_TOKENIZER)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from search_r1.plan.reader_prompt import (ReaderPromptBuilder, collate, answer_class, build_negctrl_map,
                                          mapping_sha256, DOC_MAX_TOKENS, DOCS_MAX_TOKENS)
from search_r1.plan.reader_worker import PhiScorer

TOK = AutoTokenizer.from_pretrained(os.environ.get('PLAN_TEST_TOKENIZER', 'Qwen/Qwen2.5-3B'))
FAILS = []


def check(cond, msg):
    print(f"[{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


def doc(i, n_words):
    return f'(Title: D{i}) ' + ' '.join(f'w{i}x{j}' for j in range(n_words))


def test_truncation():
    b = ReaderPromptBuilder(TOK)
    long_doc = doc(0, 400)
    block, kept = b.docs_block([long_doc])
    n_doc = len(TOK(long_doc, add_special_tokens=False)['input_ids'])
    check(n_doc > DOC_MAX_TOKENS and kept == 1 and
          len(block) == DOC_MAX_TOKENS + len(b._ids('Document 1: ')) + 2 * len(b._ids('\n')),
          f'single doc cut to {DOC_MAX_TOKENS} tokens ({n_doc} before)')

    docs = [doc(i, 300) for i in range(15)]
    block, kept = b.docs_block(docs)
    text = TOK.decode(block)
    check(len(block) <= DOCS_MAX_TOKENS and kept < 15, f'docs block {len(block)} <= {DOCS_MAX_TOKENS}, kept {kept}/15')
    check('(Title: D14)' in text and f'(Title: D{15 - kept})' in text and f'(Title: D{14 - kept})' not in text,
          'the earliest docs are the ones dropped')
    check(text.startswith('Document 1: (Title: D') and f'Document {kept}: ' in text, 'numbering restarts on kept docs')

    block1, k1 = b.docs_block([doc(1, 5), doc(2, 5), doc(1, 5)])
    block2, k2 = b.docs_block([doc(1, 5), doc(2, 5)])
    check(k1 == 2 and block1 == block2, 'identical passages are kept once')
    check(b.docs_block([]) == ([], 0), 'no docs -> empty block')


def test_build_collate():
    b = ReaderPromptBuilder(TOK)
    rows = b.build('who wrote hamlet?', [doc(1, 10)], ['William Shakespeare', ' William Shakespeare', 'Shakespeare', ''])
    check(len(rows) == 2, 'one row per distinct non-empty alias')
    ctx0 = rows[0][0][:-rows[0][1]]
    ctx1 = rows[1][0][:-rows[1][1]]
    check(ctx0 == ctx1, 'context identical across aliases')
    check(TOK.decode(rows[1][0][-rows[1][1]:]) == 'Shakespeare', 'target = alias tokens at the end')
    full = TOK.decode(ctx0)
    check('Question: who wrote hamlet?' in full and 'Document 1: (Title: D1)' in full and
          full.rstrip().endswith('assistant'), 'chat template: docs, then question, then generation prompt')
    ids, attn, pos, tm = collate(rows, TOK.pad_token_id)
    ok = True
    for i, (r, tl) in enumerate(rows):
        ok &= int(attn[i].sum()) == len(r) and int(tm[i].sum()) == tl
        ok &= ids[i][tm[i].bool()].tolist() == r[-tl:]
    check(ok, 'collate: target mask covers exactly the alias tokens, right-padded')
    return rows


def test_scorer(rows):
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(TOK), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    lm = Qwen2ForCausalLM(cfg).eval()
    scorer = PhiScorer(lm).eval()
    ids, attn, pos, tm = collate(rows, TOK.pad_token_id)
    with torch.no_grad():
        got = scorer(ids, attn, pos, tm)
        ref = []
        for i, (r, tl) in enumerate(rows):
            x = torch.tensor([r])
            lp = torch.log_softmax(lm(x).logits.float(), -1)[0]
            tgt = range(len(r) - tl, len(r))
            ref.append(sum(lp[t - 1, r[t]].item() for t in tgt) / tl)
        ref = torch.tensor(ref)
        pad = 7
        ids2 = torch.cat([ids, torch.full((ids.shape[0], pad), TOK.pad_token_id)], 1)
        z = torch.zeros(ids.shape[0], pad, dtype=torch.long)
        attn2, tm2 = torch.cat([attn, z], 1), torch.cat([tm, z], 1)
        pos2 = (attn2.cumsum(-1) - 1).clamp_min(0)
        got2 = scorer(ids2, attn2, pos2, tm2)
    check(torch.allclose(got, ref, atol=1e-4), f'PhiScorer == mean target log-prob from full logits {got.tolist()}')
    check(torch.allclose(got, got2, atol=1e-4), 'PhiScorer unchanged by extra right padding')


def test_negctrl():
    cases = {'yes': 'yesno', 'No': 'yesno', '1984': 'date', 'March 3, 1990': 'date', '12/05/2001': 'date',
             '1990s': 'date', '42': 'number', '3.5 million': 'number', 'seven': 'number', '1,200': 'number',
             'Paris': 'entity', 'The Beatles': 'entity', 'May Day': 'entity'}
    bad = {a: (answer_class([a]), c) for a, c in cases.items() if answer_class([a]) != c}
    check(not bad, f'answer classes {bad or ""}')
    q = {}
    for i in range(30):
        q[f'nq/{i}'] = [f'Entity {i}']
        q[f'hotpotqa/{i}'] = ['yes' if i % 2 else 'no']
        q[f'nq/n{i}'] = [str(100 + i)]
        q[f'nq/d{i}'] = [str(1900 + i)]
    q['nq/only_date'] = ['June 1, 1990']
    m, counts = build_negctrl_map(q, seed=0)
    ok = set(m) == set(q) and all(answer_class(q[a]) == answer_class(q[b]) for a, b in m.items())
    ok &= all(a != b for a, b in m.items() if counts[answer_class(q[a])] > 1)
    check(ok, f'negctrl map: every qid mapped within its class, no fixed points {counts}')
    check(len(set(m.values())) == len(m), 'negctrl map is a permutation')
    m2, _ = build_negctrl_map(dict(reversed(list(q.items()))), seed=0)
    check(m == m2 and mapping_sha256(m) == mapping_sha256(m2), 'deterministic for seed 0, input order irrelevant')
    m3, _ = build_negctrl_map(q, seed=1)
    check(m3 != m, 'a different seed gives a different map')


def main():
    test_truncation()
    rows = test_build_collate()
    test_scorer(rows)
    test_negctrl()
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED'}")
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
