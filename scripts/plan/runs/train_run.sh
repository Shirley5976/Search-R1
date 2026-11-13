#!/bin/bash
# One Tier A/B/C training run from theta_w (R020-R026, R030-R033, R040, R041).
#   bash scripts/plan/runs/train_run.sh <run_id> <method> <seed> [budget_gpu_h] [extra hydra overrides...]
# method: p0 | p2_all | p3 | p1:<alpha> ; append @neg for the negative-control reader (e.g. p1:0.3@neg).
# P2-all / P3 take beta_hat (and M* for P3) from prereg/pilot_result.json, never from the command line.
# Re-running the same command resumes from the latest complete checkpoint (budget clock restored).
# Evaluation: curve subsample every 10 GPU h (+ t=0); after the budget, the final checkpoint is evaluated
# on the 7 full test sets (val_only).
set -o pipefail
source "$(dirname "$0")/common.sh"

RUN_ID=$1; METHOD_SPEC=$2; SEED=$3; BUDGET=${4:-160}; shift 4 2>/dev/null || shift $#
[ -n "$RUN_ID" ] && [ -n "$METHOD_SPEC" ] && [ -n "$SEED" ] || { echo "usage: $0 run_id method seed [budget]"; exit 1; }

TARGET=main
case $METHOD_SPEC in *@neg) TARGET=negctrl; METHOD_SPEC=${METHOD_SPEC%@neg};; esac
METHOD=${METHOD_SPEC%%:*}
PLAN_ARGS=(plan.enable=true plan.method=$METHOD plan.reader_target=$TARGET)
case $METHOD in
    p0) ;;
    p2_all) PLAN_ARGS+=(plan.beta=$(pilot_get "r['rows']['p2_all|main']['beta_hat']")) ;;
    p3) M=$(pilot_get "r['p3_M_star']")
        PLAN_ARGS+=(plan.M=$M plan.beta=$(pilot_get "r['rows']['p3_m$M|main']['beta_hat']")) ;;
    p1) PLAN_ARGS+=(plan.alpha=${METHOD_SPEC#*:}) ;;
    *) echo "unknown method $METHOD"; exit 1 ;;
esac
[ "$METHOD" = p0 ] && [ "$TARGET" = main ] && PLAN_ARGS=(plan.enable=true plan.method=p0)

NAME=$(echo "$RUN_ID-$METHOD_SPEC-$TARGET-s$SEED" | tr ':' '_')
RESUME=$(latest_ckpt $NAME)
INIT_ARGS=(trainer.plan_init_path=$THETA_W)
[ -n "$RESUME" ] && INIT_ARGS=(++trainer.resume_mode=from_path ++trainer.resume_from_path=$RESUME) && echo "[resume] $RESUME"

if [ ! -f $CKPT/$NAME/.train_done ]; then
    run_ppo $NAME \
        data.train_files=$DATA_DIR/train_pool.parquet \
        trainer.default_local_dir=$CKPT/$NAME trainer.plan_eval_dir=$CKPT/$NAME/eval \
        ++trainer.val_only=false ++trainer.val_before_train=true \
        trainer.seed=$SEED \
        trainer.budget_gpu_h=$BUDGET \
        trainer.eval_every_gpu_h=10 \
        trainer.save_every_gpu_h=20 \
        trainer.save_keep_gpu_h=[80] \
        trainer.plan_save_final=true \
        "${INIT_ARGS[@]}" "${PLAN_ARGS[@]}" "$@" || exit 1
    touch $CKPT/$NAME/.train_done
fi

# final full evaluation on the 7 test sets (greedy), once
FINAL=$(latest_ckpt $NAME)
if [ ! -f $CKPT/$NAME/.final_eval_done ]; then
    run_ppo $NAME-final \
        data.train_files=$DATA_DIR/train_pool.parquet \
        data.val_files=data/nq_hotpotqa_train/test.parquet \
        trainer.default_local_dir=$CKPT/$NAME-final trainer.plan_eval_dir=$CKPT/$NAME/eval \
        ++trainer.val_only=true ++trainer.val_only_tag=final_full \
        ++trainer.resume_mode=from_path ++trainer.resume_from_path=$FINAL "$@" || exit 1
    touch $CKPT/$NAME/.final_eval_done
fi
echo "[done] $NAME -> $CKPT/$NAME/eval"
