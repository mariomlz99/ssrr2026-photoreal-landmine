#!/usr/bin/env bash
# SSRR 2026 - full eval + multi-seed variance aggregation. Run AFTER variance_study.sh finishes.
#
#   PART 1  canonical seed-42 eval -> every paper table (eval_results/ + results/ via make_results1-4)
#   PART 2  per-seed eval (seeds 42-46, headline subset) + within_rep -> mean+-std variance table
#
# Usage:  nohup bash eval_variance.sh > logs/eval_variance.log 2>&1 &
set -u
cd "$(dirname "$(readlink -f "$0")")"
[ -f configs/env.sh ] && source configs/env.sh
PY="${SSRR_PY:-python3}"
mkdir -p logs
HEADLINE="baseline h_off t_10 sun_off n_30 inv_off"   # + real (eval_real_ablation_suland)
log(){ echo ">>> [$(date +'%F %T')] $*"; }

############ PART 1 - canonical full-paper tables (seed 42) ############
log "PART 1  canonical seed-42 eval (all paper tables)"
export SSRR_RUN_SEED=42
D=eval_results/ablation_suland
for sz in l s n; do for ds in 30k 20k 10k; do
  $PY evaluation/eval_ablation_suland.py --size "$sz" --datasize "$ds" --outdir "$D/yolo11${sz}_${ds}"
done; done
for sz in l s n; do
  $PY evaluation/eval_real_ablation_suland.py --size "$sz" --outdir "$D/yolo11${sz}_30k"
done
$PY evaluation/eval_iou_sweep_conf_0.25_0.5_0.75.py --size l --datasize 30k --outdir eval_results/iou_conf_sweep/yolo11l_30k
$PY evaluation/eval_prevalence_volume.py
for i in 1 2 3 4; do $PY "analysis/make_results${i}_log.py"; done
log "PART 1 done -> tables in eval_results/ and results/"

############ PART 2 - multi-seed variance (headline subset) ############
log "PART 2  per-seed headline eval (seeds 42-46) + within_rep"
for S in 42 43 44 45 46; do
  export SSRR_RUN_SEED="$S"
  OUT="eval_results/variance/seed${S}"
  log "  seed $S -> $OUT"
  $PY evaluation/eval_ablation_suland.py     --size l --datasize 30k --configs $HEADLINE --outdir "$OUT"
  $PY evaluation/eval_real_ablation_suland.py --size l                                    --outdir "$OUT"
done
# sigma_within: 2nd seed-42 realization (separate base) for the 3 probe configs
export SSRR_RUN_SEED=42
WR="eval_results/variance/within_rep"
SSRR_RUNS_BASE="$PWD/_within_rep" $PY evaluation/eval_ablation_suland.py     --size l --datasize 30k --configs baseline t_10 --outdir "$WR"
SSRR_RUNS_BASE="$PWD/_within_rep" $PY evaluation/eval_real_ablation_suland.py --size l                                          --outdir "$WR"
log "PART 2 eval done -> aggregating"

$PY analysis/make_variance_table.py
log "=== eval_variance.sh COMPLETE -> eval_results/variance/variance_table.csv (+ printed summary) ==="
