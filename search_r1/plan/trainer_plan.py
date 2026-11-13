"""Plan additions to RayPPOTrainer (checklist items 2-9), kept out of ray_trainer.py.

- _plan_step: one training step for P0 / P1 / P2-all / P2-sel / P3 with the single-step update.
- fit_pilot: R012a pilot dump on theta_w (M siblings, all phi variants, ref log-probs; no update).
- seeding, init from theta_w, GPU-hour checkpoints (keep latest + a keep list), metrics.jsonl.
"""

import json
import os
import shutil
from contextlib import contextmanager

import numpy as np
import torch
from codetiming import Timer

from verl import DataProto
from search_r1.plan.phi import PhiEngine, collect_phis, lexical_action_phi, lexical_state_phi
from search_r1.plan.step import (METHODS, phi_needs, fork_m_for, compute_plan_advantages, build_update_batch,
                                 fork_stats, TENSOR_KEYS)
from search_r1.plan.advantages import token_advantages


@contextmanager
def _timer(name, timing_raw):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


def _qids(batch):
    nt = batch.non_tensor_batch
    return [f'{s}/{i}' for s, i in zip(nt['data_source'], nt['id'])]


def _golds(batch):
    return [list(rm['ground_truth']['target']) for rm in batch.non_tensor_batch['reward_model']]


class PlanTrainerMixin:

    # ------------------------------------------------------------------ setup
    @property
    def plan_enabled(self):
        p = self.config.get('plan', None)
        return p is not None and bool(p.get('enable', False))

    def _plan_init(self):
        """Called from RayPPOTrainer.__init__ after the config is final."""
        if not self.plan_enabled:
            return
        p = self.config.plan
        assert p.method in METHODS, p.method
        self.plan_mode = p.get('mode', 'train')
        assert self.plan_mode in ('train', 'pilot')
        self.plan_needs = phi_needs(p.method, self.plan_mode)
        self.plan_fork_m = fork_m_for(p.method, p.M, self.plan_mode)
        if p.method == 'p3' or self.plan_mode == 'pilot':
            assert self.plan_fork_m >= 1, 'P3 / pilot need plan.M >= 1'
        self.plan_use_reader = bool(self.plan_needs)
        act = self.config.actor_rollout_ref.actor
        assert act.plan_single_step and act.state_masking and act.use_kl_loss, \
            'plan runs need actor.plan_single_step, state_masking and use_kl_loss'
        ro = self.config.actor_rollout_ref.rollout
        assert float(ro.temperature) == 1.0 and float(ro.top_p) == 1.0 and int(ro.top_k) == -1
        self.plan_negctrl = None
        if p.get('reader_target', 'main') == 'negctrl' or self.plan_mode == 'pilot':
            with open(p.negctrl_map, 'r', encoding='utf-8') as f:
                self.plan_negctrl = json.load(f)

    def _plan_init_reader(self, all_wg):
        if self.plan_enabled and self.plan_use_reader:
            self.reader_wg = all_wg['reader']
            self.reader_wg.init_model()
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(self.config.plan.reader.path)
            self.phi_engine = PhiEngine(tok, self.reader_wg.compute_phi, self.reader_wg.world_size)

    def _plan_seed(self):
        s = self.config.trainer.get('seed', None)
        return None if s is None else int(s)

    def _plan_reseed_workers(self):
        s = self._plan_seed()
        if s is not None:
            self.actor_rollout_wg.plan_reseed(s)

    def _plan_gen_kwargs(self):
        if not self.plan_enabled:
            return {}
        return dict(plan_fork_m=self.plan_fork_m, plan_kstar_max=self.config.max_turns + 1,
                    plan_seed=self._plan_seed() or 0)

    # ------------------------------------------------------------------ logging / checkpoints
    def _plan_log_jsonl(self, metrics):
        d = self.config.trainer.get('plan_eval_dir', None)
        if not d:
            return
        os.makedirs(d, exist_ok=True)
        row = {'step': self.global_steps}
        for k, v in metrics.items():
            try:
                row[k] = float(v)
            except (TypeError, ValueError):
                continue
        with open(os.path.join(d, 'metrics.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(row) + '\n')

    def _plan_save_due(self, budget):
        every = float(self.config.trainer.get('save_every_gpu_h', -1) or -1)
        keep = [float(x) for x in (self.config.trainer.get('save_keep_gpu_h', None) or [])]
        used = budget.used_gpu_h
        due, mark_keep = False, False
        if every > 0:
            nxt = getattr(self, 'plan_next_save_gpu_h', None) or every
            if used >= nxt:
                due = True
                self.plan_next_save_gpu_h = (int(used // every) + 1) * every
            else:
                self.plan_next_save_gpu_h = nxt
        done = set(getattr(self, 'plan_kept_gpu_h', []))
        for k in keep:
            if used >= k and k not in done:
                due, mark_keep = True, True
                done.add(k)
        self.plan_kept_gpu_h = sorted(done)
        return due, mark_keep

    def _plan_prune_checkpoints(self, keep_path):
        """Keep the newest complete checkpoint and any marked .plan_keep; delete other global_step_*."""
        actor_dir = os.path.dirname(os.path.abspath(keep_path))
        for entry in os.scandir(actor_dir):
            if not entry.is_dir() or not entry.name.startswith('global_step_'):
                continue
            if os.path.abspath(entry.path) == os.path.abspath(keep_path):
                continue
            if os.path.exists(os.path.join(entry.path, '.plan_keep')):
                continue
            print(f'[plan] removing old checkpoint {entry.path}')
            shutil.rmtree(entry.path, ignore_errors=True)

    def _plan_after_save(self, epoch, batch_idx, mark_keep):
        path = os.path.join(self.config.trainer.default_local_dir, 'actor', f'global_step_{self.global_steps}')
        assert os.path.exists(os.path.join(path, '.checkpoint_complete')), path
        if mark_keep:
            open(os.path.join(path, '.plan_keep'), 'w').close()
        if self.config.trainer.get('save_keep_latest_only', True):
            self._plan_prune_checkpoints(path)
        return path

    def _plan_extra_state(self):
        return {'plan_next_save_gpu_h': getattr(self, 'plan_next_save_gpu_h', None),
                'plan_kept_gpu_h': getattr(self, 'plan_kept_gpu_h', [])}

    def _plan_load_extra_state(self, state):
        self.plan_next_save_gpu_h = state.get('plan_next_save_gpu_h', None)
        self.plan_kept_gpu_h = state.get('plan_kept_gpu_h', [])

    # ------------------------------------------------------------------ one step
    def _plan_reader_targets(self, batch, golds):
        if self.config.plan.get('reader_target', 'main') == 'negctrl':
            return [self.plan_negctrl[q]['aliases'] for q in _qids(batch)]
        return golds

    def _plan_rollout(self, batch, gen_batch, gm, timing_raw):
        first = gen_batch.batch['input_ids'][:, -gm.config.max_start_length:].clone().long()
        # k* draws depend only on (seed, step), so a resumed run redraws the same k*
        gm.plan_rng = np.random.RandomState([self._plan_seed() or 0, int(self.global_steps)])
        with _timer('gen', timing_raw):
            gm.timing_raw = timing_raw
            out = gm.run_llm_loop(gen_batch=gen_batch, initial_input_ids=first)
        for key in out.batch.keys():
            out.batch[key] = out.batch[key].long()
        sib = gm.last_siblings
        if sib is not None:
            for key in sib.batch.keys():
                sib.batch[key] = sib.batch[key].long()
        nt = batch.non_tensor_batch
        # index restarts at 0 in each data source, so group by data_source/index
        nt['uid'] = np.array([f'{s}/{i}' for s, i in zip(nt['data_source'], nt['index'])], dtype=object)
        batch = batch.union(out)
        return batch, sib, gm.last_sib_records, gm.last_trace, gm.last_k_star

    def _plan_empty_trajs(self, trace, k_star):
        return [dict(n_turns=len(t), k_star=int(k_star[i]), phi_prev={}, search_turns={}, fork_phis=None)
                for i, t in enumerate(trace)]

    def _plan_step(self, batch, gen_batch, gm, metrics, timing_raw):
        """Rollout -> reward -> phi -> advantages -> ref (originals) -> expanded batch -> old log-prob
        -> one actor update. Returns the batch of originals (for data metrics)."""
        from verl.trainer.ppo.ray_trainer import compute_data_metrics
        p = self.config.plan
        batch, sib, recs, trace, k_star = self._plan_rollout(batch, gen_batch, gm, timing_raw)
        n = len(batch)
        with _timer('reward', timing_raw):
            score_tok = self.reward_fn(batch)
        scores = score_tok.sum(-1).numpy()
        batch.batch['token_level_scores'] = score_tok
        golds = _golds(batch)

        trajs = self._plan_empty_trajs(trace, k_star)
        if self.plan_use_reader:
            with _timer('reader', timing_raw):
                self.phi_engine.reset()
                trajs, _ = collect_phis(self.phi_engine, trace, recs, k_star, list(batch.non_tensor_batch['question']),
                                        golds, self._plan_reader_targets(batch, golds), self.plan_needs)
            metrics.update({f'plan/reader_{k}': v for k, v in self.phi_engine.stats.items()})
        z, adj, sib_adv = compute_plan_advantages(p.method, scores, list(batch.non_tensor_batch['uid']), trajs,
                                                  float(p.beta), float(p.alpha))

        with _timer('ref', timing_raw):
            batch = batch.union(self.ref_policy_wg.compute_ref_log_prob(batch))

        ws = self.actor_rollout_wg.world_size
        upd, n_sib, n_pad = build_update_batch(batch, sib, recs, z, adj, sib_adv, max(int(p.M), 1), ws)
        with _timer('old_log_prob', timing_raw):
            upd = upd.union(self.actor_rollout_wg.compute_log_prob(upd))

        view = DataProto(batch=upd.batch[:n], non_tensor_batch={}, meta_info=batch.meta_info)
        view.batch['token_level_scores'] = score_tok
        view.batch['token_level_rewards'] = score_tok
        view.batch['returns'] = view.batch['advantages']
        metrics.update(compute_data_metrics(batch=view, use_critic=False))
        metrics.update(fork_stats(trajs, recs))
        metrics.update({'plan/n_orig': n, 'plan/n_sib': n_sib, 'plan/n_pad': n_pad,
                        'plan/truncated_prompt': float(getattr(gm, 'plan_truncated', False)),
                        'plan/zero_var_group_frac': float(np.mean([abs(v) < 1e-12 for v in z]))})

        self._balance_batch(upd, metrics=metrics)
        upd.meta_info['global_token_num'] = torch.sum(upd.batch['attention_mask'], dim=-1).tolist()
        upd.meta_info['loss_len_const'] = float(self.config.actor_rollout_ref.actor.plan_loss_len)
        upd.meta_info['temperature'] = float(self.config.actor_rollout_ref.rollout.temperature)
        with _timer('update_actor', timing_raw):
            out = self.actor_rollout_wg.update_actor(upd)
        metrics.update({k: float(np.mean(v)) for k, v in out.meta_info['metrics'].items()})
        return view

    # ------------------------------------------------------------------ pilot dump (R012a)
    def fit_pilot(self, gm):
        """One pass over data.train_files (pilot_calib + pilot_eval), no update. Per batch writes
        pilot_XXXX.pt (token tensors of originals and siblings, ref log-probs) and pilot_XXXX.json
        (trace, scores, k*, all phi variants), then a .done marker. Re-running skips done batches."""
        p = self.config.plan
        out_dir = p.pilot_dump_dir
        os.makedirs(out_dir, exist_ok=True)
        for batch_idx, batch_dict in enumerate(self.train_dataloader):
            stem = os.path.join(out_dir, f'pilot_{batch_idx:04d}')
            if os.path.exists(stem + '.done'):
                continue
            self.global_steps = batch_idx
            timing = {}
            batch = DataProto.from_single_dict(batch_dict)
            batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
            gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
            with _timer('step', timing):
                batch, sib, recs, trace, k_star = self._plan_rollout(batch, gen_batch, gm, timing)
                with _timer('reward', timing):
                    scores = self.reward_fn(batch).sum(-1).numpy()
                golds = _golds(batch)
                qids = _qids(batch)
                questions = list(batch.non_tensor_batch['question'])
                need = {'prev_all', 'fork'}
                with _timer('reader', timing):
                    self.phi_engine.reset()
                    main, _ = collect_phis(self.phi_engine, trace, recs, k_star, questions, golds, golds, need)
                    main_stats = dict(self.phi_engine.stats)
                    self.phi_engine.reset()
                    neg_t = [self.plan_negctrl[q]['aliases'] for q in qids]
                    neg, _ = collect_phis(self.phi_engine, trace, recs, k_star, questions, golds, neg_t, need)
                with _timer('ref', timing):
                    batch = batch.union(self.ref_policy_wg.compute_ref_log_prob(batch))
            records = self._pilot_records(batch, trace, recs, k_star, scores, golds, qids, main, neg)
            tensors = {'orig': {k: batch.batch[k].cpu() for k in TENSOR_KEYS + ('ref_log_prob',)},
                       'sib': None if sib is None else {k: sib.batch[k].cpu() for k in TENSOR_KEYS}}
            torch.save(tensors, stem + '.pt')
            with open(stem + '.json', 'w', encoding='utf-8') as f:
                json.dump({'batch_idx': batch_idx, 'timing_s': timing, 'reader_stats': main_stats,
                           'n_gpus': self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
                           'trajectories': records}, f, ensure_ascii=False)
            open(stem + '.done', 'w').close()
            print(f'[pilot] batch {batch_idx}: {len(records)} trajectories, '
                  f'{0 if sib is None else len(sib)} siblings, step {timing["step"]:.1f}s')

    def _pilot_records(self, batch, trace, recs, k_star, scores, golds, qids, main, neg):
        sib_rows = {}
        for s, r in enumerate(recs):
            sib_rows.setdefault(r['parent'], []).append(s)
        out = []
        for i, tr in enumerate(trace):
            K = len(tr)
            rec = {'qid': qids[i], 'uid': str(batch.non_tensor_batch['uid'][i]), 'row': i,
                   'score': float(scores[i]), 'n_turns': K, 'k_star': int(k_star[i]),
                   'actions': [r['action'] or 'invalid' for r in tr], 'executed': [bool(r['executed']) for r in tr],
                   'n_docs': [len(r['docs']) for r in tr],
                   'phi_prev': {'main': [main[i]['phi_prev'][k] for k in range(1, K + 1)],
                                'neg': [neg[i]['phi_prev'][k] for k in range(1, K + 1)],
                                'lex': [lexical_state_phi(tr, k - 1, golds[i]) for k in range(1, K + 1)]},
                   'fork': None}
            if main[i]['fork_phis'] is not None:
                k = int(k_star[i])
                cands = [tr[k - 1]] + [recs[s] for s in sib_rows[i]]
                rec['fork'] = {'sib_rows': sib_rows[i], 'actions': [c['action'] or 'invalid' for c in cands],
                               'executed': [bool(c['executed']) for c in cands],
                               'main': main[i]['fork_phis'], 'neg': neg[i]['fork_phis'],
                               'lex': [lexical_action_phi(tr, c, golds[i]) for c in cands]}
            out.append(rec)
        return out
