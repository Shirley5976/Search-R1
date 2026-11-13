# Shared settings for every plan run (sourced by the run scripts in this directory).
# Assumes 4 GPUs, the retriever on 127.0.0.1:8007 (scripts/plan/runs/retriever.sh) and cached models.

cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export DATA_DIR=${DATA_DIR:-data/prereg}
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE=${WANDB_MODE:-offline}

export WAND_PROJECT=${WAND_PROJECT:-Search-R1-plan}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-3B}
export READER_MODEL=${READER_MODEL:-Qwen/Qwen2.5-7B-Instruct}
export NGPU=${NGPU:-4}
export RETRIEVER_URL=${RETRIEVER_URL:-http://127.0.0.1:8007/retrieve}
export PY=${PY:-python3}
export CKPT=${CKPT:-verl_checkpoints}

# B is fixed by R007; export B=64 if the budget cut rule 3 applies
B=${B:-128}
G=5
# theta_w (R010): actor checkpoint dir with optimizer state, written by r010_warmup.sh
THETA_W=${THETA_W:-$CKPT/r010-warmup/actor/global_step_50}
# pilot outputs (R015): prereg/pilot_result.json gives beta_hat and M*
PILOT_RESULT=${PILOT_RESULT:-prereg/pilot_result.json}

COMMON_ARGS=(
    data.val_files=$DATA_DIR/curve_eval.parquet
    data.train_data_num=null data.val_data_num=null
    data.train_batch_size=$B data.val_batch_size=125
    data.max_prompt_length=4096 data.max_response_length=500
    data.max_start_length=2048 data.max_obs_length=500
    data.shuffle_train_dataloader=True
    algorithm.adv_estimator=grpo algorithm.no_think_rl=false
    actor_rollout_ref.model.path=$BASE_MODEL
    actor_rollout_ref.model.enable_gradient_checkpointing=true
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP:-0}
    actor_rollout_ref.actor.use_kl_loss=true
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.ppo_epochs=1
    actor_rollout_ref.actor.ppo_mini_batch_size=$((B * G))
    actor_rollout_ref.actor.ppo_micro_batch_size=${MICRO:-8}
    actor_rollout_ref.actor.fsdp_config.param_offload=true
    actor_rollout_ref.actor.fsdp_config.grad_offload=true
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
    actor_rollout_ref.actor.state_masking=true
    actor_rollout_ref.actor.plan_single_step=true
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.gpu_memory_utilization=${VLLM_MEM:-0.45}
    actor_rollout_ref.rollout.temperature=1
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.n_agent=$G
    actor_rollout_ref.ref.log_prob_micro_batch_size=4
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    trainer.logger=['console','wandb']
    trainer.default_hdfs_dir=null
    trainer.n_gpus_per_node=$NGPU trainer.nnodes=1
    trainer.save_freq=-1 trainer.test_freq=-1
    trainer.project_name=$WAND_PROJECT
    trainer.total_epochs=1
    max_turns=3
    retriever.url=$RETRIEVER_URL retriever.topk=3
    plan.reader.path=$READER_MODEL
)

mkdir -p logs

# run_ppo <experiment_name> <extra hydra overrides...>
# Keys missing from ppo_trainer.yaml (val_only, val_before_train, resume_from_path, val_only_tag) are
# passed as ++key=value. Every key once: the caller sets default_local_dir and plan_eval_dir.
run_ppo() {
    local name=$1; shift
    echo "[run] $name  $*"
    PYTHONUNBUFFERED=1 $PY -m verl.trainer.main_ppo \
        "${COMMON_ARGS[@]}" \
        trainer.experiment_name=$name \
        "$@" 2>&1 | tee -a logs/$name.log
    return ${PIPESTATUS[0]}
}

# latest complete checkpoint of a run, or empty (used to resume after a crash)
latest_ckpt() {
    local d=$CKPT/$1/actor
    [ -d "$d" ] || return 0
    ls -d $d/global_step_* 2>/dev/null | while read p; do
        [ -f "$p/.checkpoint_complete" ] && echo "${p##*_} $p"
    done | sort -n | tail -1 | cut -d' ' -f2
}

# value from prereg/pilot_result.json: pilot_get <python expression on r>
pilot_get() {
    $PY -c "import json; r = json.load(open('$PILOT_RESULT')); print($1)"
}
