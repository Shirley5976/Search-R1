#!/bin/bash
# M0 / M2 stages. Run in this order (each step is re-runnable; finished parts are skipped or resumed):
#   bash scripts/plan/runs/stages.sh r002      # vanilla-settings P0 smoke, 20 steps
#   bash scripts/plan/runs/stages.sh r004      # B0b: B=16 dump, formula path, impl on 1 GPU and 4 GPUs, compare
#   bash scripts/plan/runs/m1_profile.sh r007 / r008 / r009
#   bash scripts/plan/runs/stages.sh r010      # warm-up 50 P0 steps -> theta_w (weights + optimizer)
#   bash scripts/plan/runs/stages.sh r011      # stage 0.5: 1+5 steps per method from theta_w, nothing saved
#   bash scripts/plan/runs/stages.sh r012a     # pilot dump on the 500 pilot questions (main + negctrl phi = R013)
#   bash scripts/plan/runs/stages.sh r012b     # per-vector sketches (4 processes, one per GPU) + R014 subset
#   bash scripts/plan/runs/stages.sh r014      # sketch vs exact check
#   bash scripts/plan/runs/stages.sh r015      # costs + W table + promotion -> prereg/pilot_result.{json,md}
# After r015: commit prereg/ (git add prereg && git commit) BEFORE any Tier A run.
set -o pipefail
source "$(dirname "$0")/common.sh"

theta_w() {
    local p=$(latest_ckpt r010-warmup)
    [ -n "$p" ] || { echo "theta_w missing: run r010 first" >&2; exit 1; }
    echo $p
}

pilot_dump() {  # pilot_dump <name> <train_files> <batch> <init or ''> [extra...]
    local name=$1 files=$2 bs=$3 init=$4; shift 4
    local args=()
    [ -n "$init" ] && args+=(trainer.plan_init_path=$init)
    run_ppo $name data.train_files="$files" data.train_batch_size=$bs data.shuffle_train_dataloader=False \
        trainer.default_local_dir=$CKPT/$name trainer.plan_eval_dir=$CKPT/$name/eval \
        ++trainer.val_only=false ++trainer.val_before_train=false trainer.seed=0 \
        plan.enable=true plan.method=p3 plan.mode=pilot plan.M=4 plan.pilot_dump_dir=$CKPT/$name/dump \
        "${args[@]}" "$@"
}

case $1 in
r002)
    run_ppo r002-p0-smoke data.train_files=$DATA_DIR/train_pool.parquet \
        trainer.default_local_dir=$CKPT/r002-p0-smoke trainer.plan_eval_dir=$CKPT/r002-p0-smoke/eval \
        ++trainer.val_only=false ++trainer.val_before_train=false \
        trainer.total_training_steps=21 trainer.seed=0 plan.enable=true plan.method=p0
    ;;
r004)
    D=$CKPT/r004-b0b
    # 16 calibration questions, rollouts from the base model; M=4 siblings
    pilot_dump r004-b0b-dump "[$DATA_DIR/pilot_calib.parquet]" 16 '' data.train_data_num=16 trainer.total_training_steps=1
    DUMP=$CKPT/r004-b0b-dump/dump
    SPECS="p0 p2_all:0.5:1 p3:0.5:4 p3:0.5:1"
    EX="model.layers.0. model.layers.1. model.layers.2."
    $PY scripts/plan/r012b_pilot_sketch.py --dump_dir $DUMP --model $BASE_MODEL --out_dir $D/formula \
        --exact_questions 0 || exit 1
    for s in $SPECS; do
        IFS=: read m b M <<< "$s"
        $PY scripts/plan/r012b_pilot_sketch.py --dump_dir $DUMP --model $BASE_MODEL --out_dir $D/formula \
            --exact_prefix $EX --b0b_method $m --b0b_beta ${b:-0} --b0b_M ${M:-1} || exit 1
    done
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 --master_port 29511 scripts/plan/r004_b0b_impl.py \
        --dump_dir $DUMP --model $BASE_MODEL --out_dir $D/impl --specs $SPECS --cpu_offload || exit 1
    torchrun --nproc_per_node $NGPU --master_port 29512 scripts/plan/r004_b0b_impl.py \
        --dump_dir $DUMP --model $BASE_MODEL --out_dir $D/impl --specs $SPECS || exit 1
    $PY scripts/plan/r004_compare.py --impl_dir $D/impl --sketch_dir $D/formula --out $D/r004_report.json
    ;;
r010)
    NAME=r010-warmup
    RESUME=$(latest_ckpt $NAME)
    ARGS=()
    [ -n "$RESUME" ] && ARGS=(++trainer.resume_mode=from_path ++trainer.resume_from_path=$RESUME)
    LR_WARMUP=0.2 source "$(dirname "$0")/common.sh"
    run_ppo $NAME data.train_files=$DATA_DIR/train_pool.parquet \
        trainer.default_local_dir=$CKPT/$NAME trainer.plan_eval_dir=$CKPT/$NAME/eval \
        ++trainer.val_only=false ++trainer.val_before_train=false \
        trainer.total_training_steps=51 trainer.seed=100 trainer.save_every_gpu_h=20 trainer.plan_save_final=true \
        plan.enable=true plan.method=p0 "${ARGS[@]}"
    echo "theta_w = $(latest_ckpt $NAME)"
    ;;
r011)
    TW=$(theta_w)
    for spec in p0 p2_all p3-M1 p3-M2 p3-M4; do
        NAME=r011-$spec
        [ -f $CKPT/$NAME/eval/metrics.jsonl ] && [ $(wc -l < $CKPT/$NAME/eval/metrics.jsonl) -ge 6 ] && continue
        rm -rf $CKPT/$NAME
        case $spec in
            p0) P=(plan.method=p0) ;;
            p2_all) P=(plan.method=p2_all plan.beta=0.5) ;;
            p3-M*) P=(plan.method=p3 plan.M=${spec#p3-M} plan.beta=0.5) ;;
        esac
        # 6 steps (first dropped), no checkpoint: theta_w on disk is never touched
        run_ppo $NAME data.train_files=$DATA_DIR/train_pool.parquet \
            trainer.default_local_dir=$CKPT/$NAME trainer.plan_eval_dir=$CKPT/$NAME/eval \
            ++trainer.val_only=false ++trainer.val_before_train=false \
            trainer.total_training_steps=7 trainer.seed=0 trainer.plan_init_path=$TW \
            plan.enable=true "${P[@]}" || exit 1
    done
    ;;
r012a)
    pilot_dump r012-pilot "[$DATA_DIR/pilot_calib.parquet,$DATA_DIR/pilot_eval.parquet]" 100 $(theta_w)
    ;;
r012b)
    TW=$(theta_w)
    torchrun --nproc_per_node $NGPU --master_port 29513 scripts/plan/r012b_pilot_sketch.py \
        --dump_dir $CKPT/r012-pilot/dump --model $TW --out_dir $CKPT/r012-pilot/sketch \
        --exact_prefix model.layers.0. model.layers.1. model.layers.2. --exact_questions 20
    ;;
r014)
    $PY scripts/plan/r014_check.py --sketch_dir $CKPT/r012-pilot/sketch --out prereg/r014_report.json
    ;;
r015)
    $PY scripts/plan/r011_costs.py --ckpt $CKPT --B $B --n_gpus $NGPU --pilot_dump $CKPT/r012-pilot/dump \
        --sketch_dir $CKPT/r012-pilot/sketch --out prereg/stage05_costs.json || exit 1
    $PY scripts/plan/r015_pilot_analysis.py --sketch_dir $CKPT/r012-pilot/sketch --costs prereg/stage05_costs.json \
        --out_prefix prereg/pilot_result --device cuda
    echo "now: git add prereg && git commit -m 'R015: pilot result (prereg)'"
    ;;
*) echo "usage: $0 r002|r004|r010|r011|r012a|r012b|r014|r015"; exit 1 ;;
esac
