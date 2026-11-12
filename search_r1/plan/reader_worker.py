"""Frozen reader worker (checklist item 6): Qwen2.5-7B-Instruct, FSDP-sharded forward only.

Colocated with actor/rollout/ref in the global resource pool. Parameters live on CPU between
scoring calls (same manual offload as the verl ref/actor). compute_phi returns, per row, the mean
log-probability of the target tokens; the driver takes exp and the max over aliases.
"""

import os

import torch
import torch.distributed
import torch.nn as nn

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, offload_fsdp_param_and_grad, load_fsdp_param_and_grad


class PhiScorer(nn.Module):
    """Root FSDP module: runs the decoder, then the LM head only on positions predicting targets."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, input_ids, attention_mask, position_ids, target_mask):
        hidden = self.lm.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                               use_cache=False).last_hidden_state
        # position t predicts token t+1
        pred_mask = target_mask[:, 1:].bool()
        h = hidden[:, :-1][pred_mask]
        labels = input_ids[:, 1:][pred_mask]
        logp = torch.log_softmax(self.lm.lm_head(h).float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        counts = pred_mask.sum(-1)
        row = torch.repeat_interleave(torch.arange(input_ids.shape[0], device=input_ids.device), counts)
        sums = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.float32).index_add_(0, row, logp)
        return sums / counts.clamp_min(1)


class ReaderWorker(Worker):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='nccl')

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
        from torch.distributed.device_mesh import init_device_mesh
        from verl.utils.fs import copy_local_path_from_hdfs
        from verl.utils.fsdp_utils import init_fn

        local_path = copy_local_path_from_hdfs(self.config.path)
        lm = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.bfloat16,
                                                  attn_implementation='flash_attention_2')
        lm.requires_grad_(False)
        lm.eval()
        wrap = get_fsdp_wrap_policy(module=lm, config=None)
        mesh = init_device_mesh('cuda', mesh_shape=(torch.distributed.get_world_size(),), mesh_dim_names=['fsdp'])
        self.module = FSDP(PhiScorer(lm), param_init_fn=init_fn, use_orig_params=False, auto_wrap_policy=wrap,
                           device_id=torch.cuda.current_device(), sharding_strategy=ShardingStrategy.FULL_SHARD,
                           mixed_precision=None, sync_module_states=True, device_mesh=mesh, forward_prefetch=False)
        self.offload = bool(self.config.get('param_offload', True))
        if self.offload:
            offload_fsdp_param_and_grad(module=self.module, offload_grad=False)
        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_phi(self, data: DataProto):
        if self.offload:
            load_fsdp_param_and_grad(module=self.module, device_id=torch.cuda.current_device(), load_grad=False)
        mb = int(self.config.get('micro_batch_size', 8))
        b = data.batch
        n = b['input_ids'].shape[0]
        outs = []
        with torch.no_grad():
            # every rank runs the same number of forwards (equal chunks from the driver)
            for s in range(0, n, mb):
                sl = slice(s, s + mb)
                attn = b['attention_mask'][sl]
                T = int(attn.sum(-1).max().item())
                args = [b[k][sl][:, :T].cuda() for k in ('input_ids', 'attention_mask', 'position_ids', 'target_mask')]
                outs.append(self.module(*args).float().cpu())
        out = torch.cat(outs) if outs else torch.zeros(0)
        if self.offload:
            offload_fsdp_param_and_grad(module=self.module, offload_grad=False)
        torch.cuda.empty_cache()
        return DataProto.from_dict(tensors={'mean_logp': out})
