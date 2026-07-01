#!/usr/bin/env bash
# SSRR 2026 - variance study (TRAINING only; eval + error-bar analysis are separate steps).
#
# Order (seed-monotonic):
#   within     Phase 1  sigma_within  : seed 42, {real-L, baseline, t_10} x2 realizations  (~1 day)
#                                       -> STOP and check the gate before the long phases
#   canonical  Phase 2  full seed-42  : the entire paper at seed 42                          (~4 days)
#   between    Phase 3  sigma_between : seeds 43-46 x {real-L + 6 headline configs}           (~5 days)
#   all        = canonical + between  (run AFTER the within-gate passes)
#
# Usage:
#   nohup bash variance_study.sh within > logs/var_within.log 2>&1 &     # then ping for the gate
#   nohup bash variance_study.sh all    > logs/var_main.log   2>&1 &     # after gate passes
#
# Resumable: every trainer skips a run whose best.pt already exists.
set -u
cd "$(dirname "$(readlink -f "$0")")"
[ -f configs/env.sh ] && source configs/env.sh
PY="${SSRR_PY:-python3}"
CACHE="${SSRR_CACHE:-disk}"            # disk = tighter determinism than ram
WK="${SSRR_WORKERS:-4}"
mkdir -p logs
CANON="${SSRR_RUNS_BASE:-$PWD}"        # canonical seed-42 + 43-46 land here (<base>/runs/detect/...)
WITHIN="$PWD/_within_rep"              # 2nd seed-42 realization, separate base, for sigma_within
HEADLINE="baseline h_off t_10 sun_off n_30 inv_off"   # + real-L (trained via train_real_sizes)
log(){ echo ">>> [$(date +'%F %T')] $*"; }

phase_within(){
  log "PHASE 1  sigma_within: seed 42, {real-L, baseline, t_10} x2"
  export SSRR_RUN_SEED=42
  # realization A -> canonical base (these 3 double as their seed-42 canonical; Phase 2 skips them)
  SSRR_RUNS_BASE="$CANON"  $PY training/train_real_sizes.py --models l            --cache "$CACHE" --workers "$WK"
  SSRR_RUNS_BASE="$CANON"  $PY training/train_ablation.py   --models l --sizes 30k --only baseline t_10 --cache "$CACHE" --workers "$WK"
  # realization B -> separate base (the comparison partner)
  SSRR_RUNS_BASE="$WITHIN" $PY training/train_real_sizes.py --models l            --cache "$CACHE" --workers "$WK"
  SSRR_RUNS_BASE="$WITHIN" $PY training/train_ablation.py   --models l --sizes 30k --only baseline t_10 --cache "$CACHE" --workers "$WK"
  log "PHASE 1 DONE -> GATE: ping to eval+diff '$CANON' vs '$WITHIN' for the 3 configs (sigma_within) BEFORE running 'all'."
}

phase_canonical(){
  log "PHASE 2  full seed-42 canonical (entire paper)"
  export SSRR_RUN_SEED=42
  export SSRR_RUNS_BASE="$CANON"
  $PY training/train_ablation.py        --models l     --sizes 30k                 --cache "$CACHE" --workers "$WK"   # 14 L/30k (baseline,t_10 skip)
  $PY training/train_real_sizes.py      --models s n                               --cache "$CACHE" --workers "$WK"   # real s,n (l skips)
  $PY training/train_ablation.py        --models n s   --sizes 30k --only baseline sun_off t_10 n_0 --cache "$CACHE" --workers "$WK"
  $PY training/train_ablation.py        --models l s n --sizes 10k 20k --only baseline sun_off       --cache "$CACHE" --workers "$WK"
  $PY training/train_prevalence_volume.py --models l s n                           --cache "$CACHE" --workers "$WK"   # Sec IV-E
  log "PHASE 2 DONE -> full paper reproducible at seed 42."
}

phase_between(){
  log "PHASE 3  sigma_between: seeds 43-46 x {real-L + $HEADLINE} @ L/30k"
  export SSRR_RUNS_BASE="$CANON"
  for S in 43 44 45 46; do
    export SSRR_RUN_SEED=$S
    log "  --- seed $S ---"
    $PY training/train_real_sizes.py --models l            --cache "$CACHE" --workers "$WK"
    $PY training/train_ablation.py   --models l --sizes 30k --only $HEADLINE --cache "$CACHE" --workers "$WK"
  done
  log "PHASE 3 DONE -> ping for eval + mean+-std error-bar tables/figures."
}

case "${1:-within}" in
  within)    phase_within ;;
  canonical) phase_canonical ;;
  between)   phase_between ;;
  all)       phase_canonical; phase_between ;;
  *) echo "usage: $0 [within|canonical|between|all]"; exit 1 ;;
esac
log "=== variance_study.sh '${1:-within}' COMPLETE ==="
