#!/bin/bash
# M1 throughput / memory / retrieval profiling (R007, R008, R009). Short runs, nothing saved.
#   bash scripts/plan/runs/m1_profile.sh r007     # P0 at B in {64, 128, 256}, 6 steps each
#   bash scripts/plan/runs/m1_profile.sh r008     # P3 M=4 with the reader, B=128, 4 steps
#   bash scripts/plan/runs/m1_profile.sh r009     # summary only: retrieval time per step from r007/r008 logs
# GPU memory is sampled by nvidia-smi into logs/<run>.gpumem.csv; the summary is
# python scripts/plan/m1_summary.py (prints per-step wall clock, sections, peak memory).
set -o pipefail
source "$(dirname "$0")/common.sh"

gpumem() {  # gpumem <name> &  (stops when the parent run ends)
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 5 > logs/$1.gpumem.csv &
    echo $!
}

profile() {  # profile <name> <steps> <overrides...>
    local name=$1 steps=$2; shift 2
    local pid=$(gpumem $name)
    run_ppo $name data.train_files=$DATA_DIR/train_pool.parquet \
        trainer.default_local_dir=$CKPT/$name trainer.plan_eval_dir=$CKPT/$name/eval \
        ++trainer.val_only=false ++trainer.val_before_train=false \
        trainer.total_training_steps=$steps trainer.seed=0 "$@"
    kill $pid 2>/dev/null
}

case $1 in
    r007)
        for b in 64 128 256; do
            B=$b; source "$(dirname "$0")/common.sh"
            profile r007-p0-B$b 6 plan.enable=true plan.method=p0
        done ;;
    r008)
        profile r008-p3-M4 4 plan.enable=true plan.method=p3 plan.M=4 plan.beta=0.5 ;;
    r009) ;;
    *) echo "usage: $0 r007|r008|r009"; exit 1 ;;
esac
$PY scripts/plan/m1_summary.py --ckpt $CKPT --runs r007-p0-B64 r007-p0-B128 r007-p0-B256 r008-p3-M4
