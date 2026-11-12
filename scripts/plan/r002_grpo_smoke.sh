#!/bin/bash
# R002: vanilla Search-R1 GRPO (P0) for 20 steps with the EXPERIMENT_PLAN.md settings.
# Needs the retriever on 127.0.0.1:8007 first: `conda activate retriever && bash retrieval_launch_cpu.sh`
# (flat index on CPU, E5 encoder on GPU 0).
#
# Batch sizes: ppo_mini_batch_size / ppo_micro_batch_size count trajectories across all GPUs.
# fsdp_workers divides them by world size and multiplies by rollout.n (=1); n_agent is expanded
# earlier in ray_trainer.fit. So B=128 questions x G=5 = 640 trajectories, and mini batch 640 means
# one optimizer step per training step.

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='data/prereg'
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_ATTENTION_BACKEND=XFORMERS
# This machine cannot reach wandb.ai: log to ./wandb locally, `wandb sync` later from a networked machine.
export WANDB_MODE=${WANDB_MODE:-offline}

WAND_PROJECT='Search-R1'
export BASE_MODEL='Qwen/Qwen2.5-3B'
export EXPERIMENT_NAME=r002-p0-smoke-qwen2.5-3b

B=128
G=5

mkdir -p logs
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train_pool.parquet \
    data.val_files=$DATA_DIR/curve_eval.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=$B \
    data.val_batch_size=125 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$((B * G)) \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n_agent=$G \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.state_masking=true \
    algorithm.no_think_rl=false \
    trainer.logger=['console','wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=1 \
    trainer.total_training_steps=20 \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8007/retrieve" \
    retriever.topk=3 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log
