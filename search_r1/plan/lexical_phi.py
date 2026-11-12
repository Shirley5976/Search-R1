"""Lexical phi for retrieval actions (plan Block 3 diagnostic control).

phi = 1 if any golden-answer alias appears in the documents of the post-retrieval state S, else 0.
- Documents are cumulative: every <information> block in S (prior turns plus this turn), matching
  the reader score Phi(S), which conditions on docs(S).
- Text is normalised with qa_em.normalize_answer (same as the EM reward), and an alias matches only
  as a whole-token sequence, so "1" does not match "1990" and "he" does not match "the".
- Aliases that normalise to the empty string are skipped.
Only <information> blocks are read; the question and the policy's own text are never matched.
See prereg/decisions.md.
"""

import re

from verl.utils.reward_score.qa_em import normalize_answer

_INFO_RE = re.compile(r'<information>(.*?)</information>', re.DOTALL)


def extract_docs(state_text):
    """All <information> blocks in a trajectory prefix, in order."""
    return _INFO_RE.findall(state_text)


def _aliases(golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    out = [normalize_answer(str(a)) for a in golden_answers]
    return [a for a in out if a]


def lexical_phi(docs, golden_answers):
    """1 if any normalised alias occurs as a whole-token sequence in any doc, else 0."""
    aliases = _aliases(golden_answers)
    if not aliases:
        return 0
    for doc in docs:
        padded = f' {normalize_answer(doc)} '
        if any(f' {a} ' in padded for a in aliases):
            return 1
    return 0


def lexical_phi_from_state(state_text, golden_answers):
    """lexical_phi over every retrieved document in the post-retrieval state text."""
    return lexical_phi(extract_docs(state_text), golden_answers)
