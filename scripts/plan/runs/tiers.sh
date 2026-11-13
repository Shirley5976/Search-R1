#!/bin/bash
# M3-M5 training queue (R020-R026, R030-R033, R040, R041, R050), one 4-GPU run at a time.
#   bash scripts/plan/runs/tiers.sh a      # Tier A: P0 x3 and the pilot winner x3, interleaved (R020-R025)
#   bash scripts/plan/runs/tiers.sh b      # Tier B: P1 screening + continuation (R030-R033), P1 under negctrl (R040)
#   bash scripts/plan/runs/tiers.sh c      # Tier C (optional): third unbiased method (R026), winner under negctrl (R041), W drift (R050)
#   bash scripts/plan/runs/tiers.sh r060   # endpoints, verdicts, interpretation table -> prereg/results.{json,md}
# Budget cut order (EXPERIMENT_PLAN): drop R041 first, then R026, then B=64.
# If the pilot winner is P0, Tier A runs P0 x3 and P3 x3 (P3 is then the "winner" slot, flagged in results).
set -o pipefail
HERE="$(dirname "$0")"
source $HERE/common.sh
[ -f $PILOT_RESULT ] || { echo "missing $PILOT_RESULT: run stages.sh r015 and commit prereg/ first"; exit 1; }
WIN=$(pilot_get "r['winner']")
SLOT=$WIN; [ "$WIN" = p0 ] && SLOT=p3
OTHER=p2_all; [ "$SLOT" = p2_all ] && OTHER=p3

case $1 in
a)
    for s in 1 2 3; do
        bash $HERE/train_run.sh R02$((s - 1)) p0 $s || exit 1
        bash $HERE/train_run.sh R02$((s + 2)) $SLOT $s || exit 1
    done ;;
b)
    bash $HERE/p1_screen.sh || exit 1
    ALPHA=$($PY -c "import json; print('%g' % json.load(open('prereg/p1_alpha.json'))['alpha'])")
    bash $HERE/train_run.sh R040 p1:$ALPHA@neg 1 ;;
c)
    bash $HERE/train_run.sh R026 $OTHER 1 || exit 1
    bash $HERE/train_run.sh R041 $SLOT@neg 1 || exit 1
    bash $HERE/w_drift.sh ;;
r060)
    $PY scripts/plan/curves.py --ckpt $CKPT --out_prefix prereg/results ;;
*) echo "usage: $0 a|b|c|r060"; exit 1 ;;
esac
