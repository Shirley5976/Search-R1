"""R004 (B0b), implementation path: the real FSDP actor's single-step gradient on a saved B=16 batch.

Builds the actor exactly as fsdp_workers.ActorRolloutRefWorker (role 'actor', no vLLM), turns the
pilot-format dump (R012a with B=16, plan.M=4) into the update batch of every requested method with
search_r1.plan.b0b (advantages.py + step.py), runs dp_actor.update_policy_single_step and captures,
through pre_step_hook (before clipping), the count-sketch of the full gradient and the full fp32
gradients of the --exact_prefix parameters. The optimizer step is disabled, so all specs see the
same weights.

  torchrun --nproc_per_node 4 scripts/plan/r004_b0b_impl.py --dump_dir D --model THETA --out_dir O
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 scripts/plan/r004_b0b_impl.py ... --cpu_offload

Output (rank 0): O/impl_<tag>_ws<world>.pt = {sketch (D,), exact {name: tensor}, n_rows, n_orig, spec}
with tag = <method>_b<beta>_m<M> (same tag as r012b --b0b_method). Compare with r004_compare.py.
"""

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

from omegaconf import OmegaConf, open_dict

from search_r1.plan import b0b
from search_r1.plan.sketch import D_DEFAULT, fsdp_full_grads, fsdp_grad_sketch, clean_name

DEFAULT_SPECS = ['p0', 'p2_all:0.5:1', 'p3:0.5:4', 'p3:0.5:1']


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump_dir', required=True)
    ap.add_argument('--model', required=True, help='HF checkpoint of theta_w')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--specs', nargs='*', default=DEFAULT_SPECS, help='method[:beta[:M]]')
    ap.add_argument('--D', type=int, default=D_DEFAULT)
    ap.add_argument('--sketch_seed', type=int, default=0)
    ap.add_argument('--exact_prefix', nargs='*', default=['model.layers.0.', 'model.layers.1.', 'model.layers.2.'])
    ap.add_argument('--kl_coef', type=float, default=0.001)
    ap.add_argument('--loss_len', type=float, default=500.0)
    ap.add_argument('--micro_batch', type=int, default=2, help='rows per micro batch per GPU')
    ap.add_argument('--cpu_offload', action='store_true', help='FSDP CPUOffload (1-GPU reference run)')
    ap.add_argument('--no_remove_padding', action='store_true')
    return ap.parse_args()


def actor_config(a, world):
    full = OmegaConf.load(os.path.join(ROOT, 'verl', 'trainer', 'config', 'ppo_trainer.yaml'))
    with open_dict(full):
        full.data.max_response_length = int(a.loss_len)
    cfg = OmegaConf.create(OmegaConf.to_container(full, resolve=True)).actor_rollout_ref
    with open_dict(cfg):
        cfg.model.path = a.model
        cfg.model.enable_gradient_checkpointing = True
        cfg.model.use_remove_padding = not a.no_remove_padding
        act = cfg.actor
        act.plan_single_step = True
        act.plan_loss_len = a.loss_len
        act.state_masking = True
        act.use_kl_loss = True
        act.kl_loss_coef = a.kl_coef
        act.kl_loss_type = 'low_var_kl'
        act.entropy_coeff = 0
        act.ppo_epochs = 1
        act.use_dynamic_bsz = False
        # fsdp_workers divides these by world size (and multiplies by rollout.n = 1)
        act.ppo_micro_batch_size = a.micro_batch * world
        act.ppo_mini_batch_size = a.micro_batch * world
        act.optim.total_training_steps = 1
        act.fsdp_config.param_offload = False
        act.fsdp_config.grad_offload = False
        act.fsdp_config.optimizer_offload = False
        act.fsdp_config.cpu_offload = bool(a.cpu_offload)
        cfg.rollout.n = 1
        cfg.rollout.log_prob_micro_batch_size = a.micro_batch * world
    return cfg


def main():
    a = parse()
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29511')
    world, rank = int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('RANK', 0))
    os.environ.setdefault('WORLD_SIZE', str(world))
    os.environ.setdefault('RANK', str(rank))
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', rank)))
    from verl.workers.fsdp_workers import ActorRolloutRefWorker

    worker = ActorRolloutRefWorker(actor_config(a, world), role='actor')
    worker.init_model()
    actor = worker.actor
    actor.actor_optimizer.step = lambda *args, **kw: None  # gradients only; weights stay theta_w

    captured = {}

    def hook(module):
        captured['sketch'] = fsdp_grad_sketch(module, a.D, a.sketch_seed).cpu()
        if a.exact_prefix:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            want = set()
            for m in FSDP.fsdp_modules(module):
                if m._handle is not None:
                    pre = [n for n, x in module.named_modules() if x is m][0]
                    pre = (pre + '.') if pre else ''
                    for fqn in m._handle.flat_param._fqns:
                        n = clean_name(pre + fqn)
                        if any(n.startswith(p) for p in a.exact_prefix):
                            want.add(n)
            captured['exact'] = fsdp_full_grads(module, sorted(want))

    actor.pre_step_hook = hook
    batches = b0b.load_dump(a.dump_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    for spec in a.specs:
        method, beta, M = b0b.parse_spec(spec)
        full = b0b.update_batch(batches, method, beta, M, world)
        local = full.chunk(world)[rank]
        local.meta_info.update({'temperature': 1.0, 'micro_batch_size': a.micro_batch, 'use_dynamic_bsz': False,
                                'loss_len_const': a.loss_len})
        local.batch = local.batch.cuda()
        # on-policy: old log-probs are the current policy's, so ratio == 1 and only its gradient survives
        local.batch['old_log_probs'] = actor.compute_log_prob(local)
        captured.clear()
        metrics = actor.update_policy_single_step(local)
        if rank == 0:
            tag = b0b.spec_tag(method, beta, M)
            out = {'spec': spec, 'tag': tag, 'world_size': world, 'cpu_offload': a.cpu_offload,
                   'sketch': captured['sketch'], 'exact': captured.get('exact', {}), 'n_rows': len(full),
                   'n_orig': float(full.batch['is_original'].sum()), 'D': a.D, 'sketch_seed': a.sketch_seed,
                   'kl_coef': a.kl_coef, 'loss_len': a.loss_len,
                   'grad_norm': metrics['actor/grad_norm'], 'n_orig_global': metrics['actor/n_orig_global']}
            torch.save(out, os.path.join(a.out_dir, f'impl_{tag}_ws{world}.pt'))
            print(f'[R004 impl] {spec}: {len(full)} rows, n_orig {out["n_orig"]:.0f}, '
                  f'grad_norm {metrics["actor/grad_norm"]}')
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == '__main__':
    main()
