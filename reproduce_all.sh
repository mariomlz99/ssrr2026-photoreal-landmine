#!/usr/bin/env bash
# =============================================================================
# SSRR 2026 -- full from-scratch reproduction of every table-feeding model.
#
#   Trains  (43 runs, ~5 days on one RTX 5090) -> evaluates -> rebuilds tables.
#   * RESUMABLE: every run is skipped if its best.pt already exists, so a crash
#     or a Ctrl-C never loses completed work -- just re-run this script.
#   * cache=ram: the image cache lives in RAM and is freed after each run, so the
#     dataset folders never accumulate .npy disk caches across the 43 runs.
#   * Priority order: Real -> L/30k (Tables III/IV) -> n,s/30k (Table V)
#     -> 10k,20k (Table VI) -> Sec IV-E controls. The headline finishes first.
#
# Usage:   nohup bash reproduce_all.sh > logs/reproduce_master.log 2>&1 &
#          tail -f logs/reproduce_master.log
# =============================================================================
set -u
cd "$(dirname "$(readlink -f "$0")")"
[ -f configs/env.sh ] && source configs/env.sh   # optional: overrides the repo-relative defaults

PY="${SSRR_PY:-python3}"            # override with SSRR_PY=/path/to/venv/bin/python3 if needed
CACHE="${SSRR_CACHE:-disk}"         # disk = portable default; SSRR_CACHE=ram is faster if you have ~36 GB RAM/run free
WORKERS="${SSRR_WORKERS:-4}"
mkdir -p logs

# init weights in CWD so Ultralytics doesn't re-download each run
for w in yolo11l.pt yolo11s.pt yolo11n.pt; do
  [ -e "$w" ] || ln -s "$SSRR_DATASETS/../$w" "$w" 2>/dev/null || true
done

stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
phase(){ echo; echo "############################################################"; \
         echo "###  [$(stamp)]  $*"; \
         echo "############################################################"; }
run(){ echo; echo ">>> [$(stamp)] $*"; "$@"; echo "<<< [$(stamp)] rc=$?"; }

echo "============================================================"
echo "  SSRR reproduce_all  --  started $(stamp)"
echo "  runs   -> $SSRR_RUNS_BASE/runs/detect"
echo "  evals  -> $SSRR_RUNS_BASE/eval_results"
echo "  data   -> $SSRR_DATA  |  synth -> $SSRR_DATASETS"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || true

# ============================ TRAINING ======================================
phase "TRAIN 1/5  Real baselines (l, s, n)"
run $PY training/train_real_sizes.py --models l s n

phase "TRAIN 2/5  14 synthetic configs at L/30k  (Tables III, IV)"
run $PY training/train_ablation.py --models l --sizes 30k --cache $CACHE --workers $WORKERS

phase "TRAIN 3/5  n,s at 30k for the 4 strongest configs  (Table V)"
run $PY training/train_ablation.py --models n s --sizes 30k --only baseline sun_off t_10 n_0 --cache $CACHE --workers $WORKERS

phase "TRAIN 4/5  Baseline + Sun-Off at l,s,n x 10k,20k  (Table VI)"
run $PY training/train_ablation.py --models l s n --sizes 10k 20k --only baseline sun_off --cache $CACHE --workers $WORKERS

phase "TRAIN 5/5  Sec IV-E class-balance / volume controls (l, s, n)"
run $PY training/train_prevalence_volume.py --models l s n --cache $CACHE --workers $WORKERS

# ============================ EVALUATION ====================================
phase "EVAL  synthetic ablation (size x datasize -> skips untrained configs)"
D=eval_results/ablation_suland
for sz in l s n; do for ds in 30k 20k 10k; do
  run $PY evaluation/eval_ablation_suland.py --size $sz --datasize $ds --outdir $D/yolo11${sz}_${ds}
done; done

phase "EVAL  real baselines (l, s, n) -> written into the 30k dirs"
for sz in l s n; do
  run $PY evaluation/eval_real_ablation_suland.py --size $sz --outdir $D/yolo11${sz}_30k
done

phase "EVAL  IoU sweep (L, 30k)  (Table IV / Fig 3)"
run $PY evaluation/eval_iou_sweep_conf_0.25_0.5_0.75.py --size l --datasize 30k --outdir eval_results/iou_conf_sweep/yolo11l_30k

phase "EVAL  Sec IV-E class-balance / volume controls"
run $PY evaluation/eval_prevalence_volume.py

# ============================ TABLES ========================================
phase "TABLES  rebuild RESULTS_1..4 from the fresh eval CSVs"
run $PY analysis/make_results1_log.py
run $PY analysis/make_results2_log.py
run $PY analysis/make_results3_log.py
run $PY analysis/make_results4_log.py

echo
echo "============================================================"
echo "  SSRR reproduce_all  --  FINISHED $(stamp)"
echo "  tables -> $SSRR_RUNS_BASE/results/RESULTS_*.txt"
echo "  Sec IV-E -> $SSRR_RUNS_BASE/eval_results/prevalence_volume/"
echo "============================================================"
