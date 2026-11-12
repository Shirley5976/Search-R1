# Pre-registered implementation decisions

Locked before any P1/P2/P3 training run. Changing any entry requires a dated note here.

## Loss normalisation (single-step update, B0c)

- Date: 2026-11-12
- Loss = sum over response tokens of (-A * ratio * loss_mask * loss_weight) / (N_orig_global * L).
- N_orig_global: number of original trajectories in the step, all-reduced over ranks (B=128, G=5 -> 640).
  Sibling trajectories (P3) do not count toward N_orig; they enter with loss_weight = 1/M.
- L = 500 = data.max_response_length, a fixed constant (config: actor.plan_loss_len).
  Chosen so the gradient scale stays close to the original token-mean loss and lr=1e-6 / grad_clip=1.0
  can be reused without retuning. It is a constant, so it does not affect unbiasedness.
- Local loss is multiplied by world_size because FSDP averages gradients across ranks (to be confirmed by B0b / R004).
- KL loss applies to original trajectories only; entropy_coeff = 0.
- Enabled with actor.plan_single_step=True; requires ppo_epochs=1, temperature=1 (asserted).

## Lexical phi (Block 3 diagnostic, R006)

- Date: 2026-11-12. Code: search_r1/plan/lexical_phi.py, test: scripts/plan/r006_lexical_test.py.
- Retrieval-action phi = 1 if any golden alias appears in the docs of the post-retrieval state S, else 0.
  Answer-action phi stays EM.
- Docs are cumulative (all <information> blocks in S), matching the reader score Phi(S) = f(docs(S)).
- "Substring match" in the plan is made precise as whole-token matching after qa_em.normalize_answer:
  plain substring would let short answers ("1", "he") match inside longer tokens ("1990", "the").
- Aliases that normalise to "" are dropped; a question with no usable alias gets phi = 0.
