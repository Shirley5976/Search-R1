"""Reader inputs for Phi(S) and the negative-control targets (checklist item 6).

Phi(S) = exp(mean_t log p_reader(y_t | q, docs(S), y_<t)), max over gold aliases.
Execution constants (FINAL_PROPOSAL.md): question + retrieved docs in retrieval order, each doc cut
to 256 reader tokens, the docs block cut to 2,048 tokens by dropping the earliest docs.
Further fixed choices (prereg/decisions.md): identical passages are kept once (first retrieval);
the prompt is assembled at token level so truncation never re-tokenises text.
"""

import hashlib
import json
import re

import numpy as np

from verl.utils.reward_score.qa_em import normalize_answer

DOC_MAX_TOKENS = 256
DOCS_MAX_TOKENS = 2048
INSTRUCTION = 'Answer the question based on the documents. Reply with only the answer.\n\n'
_MARK = '\x00DOCS\x00'


def dedup_docs(docs):
    seen, out = set(), []
    for d in docs:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def clean_aliases(targets):
    seen, out = set(), []
    for a in list(targets):
        a = str(a).strip()
        if a and a not in seen:
            seen.add(a)
            out.append(a)
    return out


class ReaderPromptBuilder:

    def __init__(self, tokenizer, doc_max=DOC_MAX_TOKENS, docs_max=DOCS_MAX_TOKENS):
        self.tok = tokenizer
        self.doc_max = doc_max
        self.docs_max = docs_max
        self._cache = {}

    def _ids(self, text):
        if text not in self._cache:
            self._cache[text] = self.tok(text, add_special_tokens=False)['input_ids']
        return self._cache[text]

    def _template(self, question):
        content = INSTRUCTION + _MARK + 'Question: ' + question
        text = self.tok.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                            add_generation_prompt=True)
        pre, post = text.split(_MARK)
        return pre, post

    def docs_block(self, docs):
        """Token ids of the docs block after per-doc and total truncation (earliest docs dropped)."""
        docs = dedup_docs(docs)
        pieces = [self.tok(d, add_special_tokens=False)['input_ids'][:self.doc_max] for d in docs]
        while pieces:
            block = []
            for n, p in enumerate(pieces, 1):
                block += self._ids(f'Document {n}: ') + p + self._ids('\n')
            block += self._ids('\n')
            if len(block) <= self.docs_max:
                return block, len(pieces)
            pieces = pieces[1:]
        return [], 0

    def build(self, question, docs, aliases, info=None):
        """One row per alias: (input_ids, target_len). The target is the last target_len tokens.
        If info is a dict it receives n_docs (after dedup) and n_kept (after truncation)."""
        pre, post = self._template(question)
        block, kept = self.docs_block(docs)
        if info is not None:
            info['n_docs'] = len(dedup_docs(docs))
            info['n_kept'] = kept
        ctx = self._ids(pre) + block + self._ids(post)
        rows = []
        for a in clean_aliases(aliases):
            t = self.tok(a, add_special_tokens=False)['input_ids']
            if t:
                rows.append((ctx + t, len(t)))
        return rows


def collate(rows, pad_id):
    """rows: list of (input_ids, target_len) -> right-padded tensors and a target mask aligned with
    input_ids (1 on target tokens)."""
    import torch
    n = len(rows)
    T = max(len(r[0]) for r in rows)
    input_ids = torch.full((n, T), pad_id, dtype=torch.long)
    attn = torch.zeros((n, T), dtype=torch.long)
    tmask = torch.zeros((n, T), dtype=torch.long)
    for i, (ids, tl) in enumerate(rows):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attn[i, :len(ids)] = 1
        tmask[i, len(ids) - tl:len(ids)] = 1
    pos = (attn.cumsum(-1) - 1).clamp_min(0)
    return input_ids, attn, pos, tmask


# ----------------------------- negative-control targets -----------------------------

_MONTHS = ('january|february|march|april|may|june|july|august|september|october|november|december|'
           'jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec')
_MONTH_RE = re.compile(rf'\b({_MONTHS})\b')
_YEAR_RE = re.compile(r'^(1\d{3}|20\d{2})s?$')
_SLASH_DATE_RE = re.compile(r'^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$')
_NUM_RE = re.compile(r'^[-+]?(\d[\d,]*)(\.\d+)?( (hundred|thousand|million|billion|trillion|percent))?$')
_NUM_WORDS = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
              'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen',
              'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety', 'hundred',
              'thousand', 'million', 'billion'}


def answer_class(aliases):
    """Fixed rule on the first non-empty alias: yes/no, then date, then number, else entity."""
    first = next((a for a in clean_aliases(aliases)), '')
    raw = first.lower().strip()
    norm = normalize_answer(first)
    if norm in ('yes', 'no'):
        return 'yesno'
    if _YEAR_RE.match(raw.rstrip(".")) or _SLASH_DATE_RE.match(raw) or (_MONTH_RE.search(norm) and re.search(r'\d', norm)):
        return 'date'
    if _NUM_RE.match(raw) or (norm and all(w in _NUM_WORDS for w in norm.split())):
        return 'number'
    return 'entity'


def build_negctrl_map(qid_to_aliases, seed=0):
    """Within each answer class, a seed-0 random cyclic permutation of the sorted qids: every
    question scores another question's gold answers (a derangement whenever the class has >1 item)."""
    classes = {}
    for qid in sorted(qid_to_aliases):
        classes.setdefault(answer_class(qid_to_aliases[qid]), []).append(qid)
    rng = np.random.RandomState(seed)
    mapping = {}
    for cls in sorted(classes):
        ids = classes[cls]
        order = rng.permutation(len(ids))
        for a, b in zip(order, np.roll(order, -1)):
            mapping[ids[a]] = ids[b]
    counts = {c: len(v) for c, v in sorted(classes.items())}
    return mapping, counts


def mapping_sha256(mapping):
    blob = json.dumps(sorted(mapping.items()), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()
