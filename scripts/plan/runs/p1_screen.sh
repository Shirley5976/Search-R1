#!/bin/bash
# M4 P1 (R030-R033): alpha in {0.1, 0.3, 1}, each for 1/3 of the budget; alpha chosen on the 1,000-question
# validation set only (mean EM); the chosen run then continues to the full budget (resume, clock restored).
#   bash scripts/plan/runs/p1_screen.sh
# The chosen alpha goes to prereg/p1_alpha.json (R040 reads it: train_run.sh R040 p1:<alpha>@neg 1).
set -o pipefail
source "$(dirname "$0")/common.sh"
SEED=${SEED:-1}
SCREEN=53.3333
ALPHAS=(0.1 0.3 1)
IDS=(R030 R031 R032)
HERE="$(dirname "$0")"

RUNS=()
for n in 0 1 2; do
    a=${ALPHAS[$n]}; NAME=${IDS[$n]}-p1_$a-main-s$SEED; RUNS+=($NAME)
    [ -f $CKPT/$NAME/.screen_done ] && continue
    mkdir -p $CKPT/$NAME
    # no full-test eval for screening runs; the validation eval below replaces it
    [ -f $CKPT/$NAME/.train_done ] || touch $CKPT/$NAME/.final_eval_done
    bash $HERE/train_run.sh ${IDS[$n]} p1:$a $SEED $SCREEN || exit 1
    run_ppo $NAME-val data.train_files=$DATA_DIR/train_pool.parquet data.val_files=$DATA_DIR/val.parquet \
        trainer.default_local_dir=$CKPT/$NAME-val trainer.plan_eval_dir=$CKPT/$NAME/eval \
        ++trainer.val_only=true ++trainer.val_only_tag=p1_screen_val \
        ++trainer.resume_mode=from_path ++trainer.resume_from_path=$(latest_ckpt $NAME) || exit 1
    touch $CKPT/$NAME/.screen_done
done

$PY scripts/plan/p1_select.py --ckpt $CKPT --runs ${RUNS[@]} --alphas ${ALPHAS[@]} --out prereg/p1_alpha.json || exit 1
BEST=$($PY -c "import json; print('%g' % json.load(open('prereg/p1_alpha.json'))['alpha'])")
for n in 0 1 2; do [ "${ALPHAS[$n]}" = "$BEST" ] && ID=${IDS[$n]}; done
NAME=$ID-p1_$BEST-main-s$SEED
echo "[P1] alpha* = $BEST; R033 = $NAME continued to 160 GPU h"
rm -f $CKPT/$NAME/.train_done $CKPT/$NAME/.final_eval_done
bash $HERE/train_run.sh $ID p1:$BEST $SEED 160
