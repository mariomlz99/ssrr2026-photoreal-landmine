#!/usr/bin/env python3
"""
SSRR 2026 -- Section IV-E: class-balance / volume control training.

Trains YOLO11 (l/s/n) on the two 5,304-image controlled subsets of Baseline-30k:
  baseline_iddbal_30k    -- matches the REAL model's class balance (23% mine-bearing,
                            ~1,455 mines): the prevalence control.
  baseline_volctrl_5304  -- matches Baseline prevalence (85% positive, ~5,346 mines)
                            at the SAME image budget: the volume control.

Both reuse the Baseline-30k validation terrains (their data.yaml points val at
baseline_30k/val). Protocol is identical to train_ablation.py (AdamW lr0 3e-4,
lrf 0.01, cos_lr, wd 5e-4, 100 ep, patience 25, batch 32, imgsz 640, seed 42 (override via SSRR_RUN_SEED), and
the sim-to-real Albumentations recipe). Only the training-set composition differs.

Weights land in  $SSRR_RUNS_BASE/runs/detect/synthetic_ablation/
  {baseline_iddbal_30k,baseline_volctrl_5304}_yolo11{l,s,n}_adamw_100ep/weights/best.pt
which is exactly where eval_prevalence_volume.py looks. Completed runs are skipped.

Usage:
    ./train_prevalence_volume.py --models l s n
    ./train_prevalence_volume.py --models l s n --cache ram --workers 4
"""
import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `import train_ablation` works
import train_ablation as T          # reuses the sim-to-real Albumentations monkeypatch,
import torch                        # OPTIMIZER_CFG, AUGMENTATION_PARAMS, SEED, RUNS_DIR, DATASETS
from ultralytics import YOLO

SETS = ["baseline_iddbal_30k", "baseline_volctrl_5304"]
RUNS_DIR = T.RUNS_DIR               # runs/detect/synthetic_ablation
DATASETS = T.DATASETS


def run_name_for(ds, m):
    base = f"{ds}_yolo11{m}_adamw_100ep"
    return base if T._RUN_SEED == T.SEED else f"{base}_seed{T._RUN_SEED}"


def is_complete(ds, m):
    return (RUNS_DIR / run_name_for(ds, m) / "weights" / "best.pt").exists()


def train_one(ds, m, args):
    rname     = run_name_for(ds, m)
    yaml_path = DATASETS / ds / "data.yaml"     # use the EXISTING yaml (shared val terrains)
    if not yaml_path.exists():
        print(f"  [ERROR] data.yaml not found for {ds}: {yaml_path}")
        return
    dropout   = 0.1 if m == "l" else 0.05
    cache_val = False if args.cache == "False" else args.cache

    print(f"\n{'#'*64}\n  RUN : {rname}\n  DATA: {yaml_path}\n"
          f"  GPU : {args.device}  batch={args.batch}  cache={cache_val}  workers={args.workers}\n{'#'*64}\n")
    model = YOLO(f"yolo11{m}.pt")
    try:
        model.train(
            data=str(yaml_path), epochs=args.epochs, patience=args.patience,
            imgsz=args.imgsz, batch=args.batch,
            optimizer=T.OPTIMIZER_CFG["name"], lr0=T.OPTIMIZER_CFG["lr0"],
            lrf=0.01, cos_lr=True, weight_decay=0.0005, dropout=dropout, seed=T._RUN_SEED,
            project=str(RUNS_DIR), name=rname, device=args.device,
            workers=args.workers, cache=cache_val, exist_ok=True,
            **T.AUGMENTATION_PARAMS,
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description="SSRR Sec IV-E class-balance/volume control training")
    ap.add_argument("--models",  nargs="+", default=["l", "s", "n"], choices=["l", "s", "n"])
    ap.add_argument("--device",  default="0")
    ap.add_argument("--epochs",  type=int, default=100)
    ap.add_argument("--patience",type=int, default=25)
    ap.add_argument("--imgsz",   type=int, default=640)
    ap.add_argument("--batch",   type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache",   default="disk", choices=["disk", "ram", "False"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    queue   = [(ds, m) for m in args.models for ds in SETS]
    pending = [(ds, m) for ds, m in queue if not is_complete(ds, m)]
    print(f"\n[prevalence/volume] {len(queue)} runs total, "
          f"{len(queue)-len(pending)} done, {len(pending)} pending")
    if args.dry_run:
        for ds, m in pending:
            print("  todo:", run_name_for(ds, m))
        return

    for ds, m in pending:
        t0 = time.time()
        try:
            train_one(ds, m, args)
            print(f"  DONE {run_name_for(ds, m)} in {(time.time()-t0)/60:.1f} min")
        except Exception as e:
            print(f"  [ERROR] {run_name_for(ds, m)}: {e}")
    print("[prevalence/volume] all done")


if __name__ == "__main__":
    main()
