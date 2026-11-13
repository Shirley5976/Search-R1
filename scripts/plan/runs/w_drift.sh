#!/bin/bash
# R050 W drift (diagnostic only, never changes the preregistration): re-measure W on 100 calibration
# questions at the t = 80 and t = 160 checkpoints of P0 seed 1 (R020) and winner seed 1 (R023).
# Pilot dump + per-vector sketches from that checkpoint, then r015 with the frozen stage-0.5 costs.
set -o pipefail
HERE="$(dirname "$0")"
source $HERE/common.sh
WIN=$(pilot_get "r['winner']"); [ "$WIN" = p0 ] && WIN=p3
for run in R020-p0-main-s1 R023-$WIN-main-s1; do
    for t in 80 160; do
        if [ $t = 80 ]; then
            CK=$(ls $CKPT/$run/actor/global_step_*/.plan_keep 2>/dev/null | head -1 | xargs -r dirname)
        else
            CK=$(latest_ckpt $run)
        fi
        [ -n "$CK" ] || { echo "no t=$t checkpoint for $run"; continue; }
        NAME=r050-$run-t$t
        if [ ! -d $CKPT/$NAME/dump ] || [ -z "$(ls $CKPT/$NAME/dump/*.done 2>/dev/null)" ]; then
            run_ppo $NAME data.train_files="[$DATA_DIR/pilot_calib.parquet]" data.train_batch_size=100 \
                data.train_data_num=100 data.shuffle_train_dataloader=False trainer.total_training_steps=1 \
                trainer.default_local_dir=$CKPT/$NAME trainer.plan_eval_dir=$CKPT/$NAME/eval \
                ++trainer.val_only=false ++trainer.val_before_train=false trainer.seed=0 trainer.plan_init_path=$CK \
                plan.enable=true plan.method=p3 plan.mode=pilot plan.M=4 plan.pilot_dump_dir=$CKPT/$NAME/dump || exit 1
        fi
        torchrun --nproc_per_node $NGPU --master_port 29514 scripts/plan/r012b_pilot_sketch.py \
            --dump_dir $CKPT/$NAME/dump --model $CK --out_dir $CKPT/$NAME/sketch --exact_prefix || exit 1
        # all 100 questions are calibration ids; r015 needs both splits, so W is read from the calib columns
        $PY scripts/plan/r015_pilot_analysis.py --sketch_dir $CKPT/$NAME/sketch --costs prereg/stage05_costs.json \
            --out_prefix $CKPT/$NAME/w_drift --n_boot 500 --split_by_half || exit 1
    done
done
