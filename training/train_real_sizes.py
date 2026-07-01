#!/usr/bin/env python3
"""
SSRR 2026 -- Real-data deployment ablation (YOLO11 n / s).

Trains the real IDD baseline at smaller model sizes (n, s) using EXACTLY the
same protocol as train_real_baseline.py (the yolo11l real model), so the real
row of the deployment ablation is directly comparable to the synthetic n/s runs
produced by train_ablation.py --models s n.

Consistency with train_real_baseline.py:
  - AdamW, lr0=3e-4, lrf=0.01, cos_lr=True, weight_decay=5e-4
  - epochs=100, patience=25, imgsz=640, batch=32, seed=42 (override via SSRR_RUN_SEED)
  - standard YOLO augments (hsv/translate/scale/fliplr/mosaic); NO albumentations
    override (real data needs no domain-gap aug)
  - dropout = 0.05 for n/s  (matches train_ablation.py's size rule; L used 0.1)
  - train split = ITA train, val split = ITA val  (val_on_val protocol)

Run names match the existing real_vv model so eval scripts line up:
  real_idd_yolo11{n,s}_adamw_100ep

Usage:
    ./train_real_sizes.py --models n s
    ./train_real_sizes.py --models n s --dry-run
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT      = Path(os.environ.get("SSRR_RUNS_BASE",str(Path(__file__).resolve().parents[1])))
RUNS_DIR  = ROOT / "runs" / "detect" / "real_ablation"
DATA_YAML = Path(os.environ.get("SSRR_DATA",str(Path(__file__).resolve().parents[1] / "SULAND"))) / "iid" / "ITA.yolo" / "ITA_train_val_2.yaml"

# Run seed: single source of truth. Override for multi-seed runs:
#   SSRR_RUN_SEED=43 python training/train_real_sizes.py --models l
_RUN_SEED = int(os.environ.get("SSRR_RUN_SEED", 42))

# ---------------------------------------------------------------------------
# Hyperparameters -- identical to train_real_baseline.py (size-dependent dropout)
# ---------------------------------------------------------------------------
TRAIN_CFG = dict(
    epochs       = 100,
    patience     = 25,
    imgsz        = 640,
    batch        = 32,
    optimizer    = "AdamW",
    lr0          = 3e-4,
    lrf          = 0.01,
    cos_lr       = True,
    weight_decay = 0.0005,
    seed         = _RUN_SEED,
    device       = "0",
    workers      = 12,
    cache        = True,
    exist_ok     = True,
)

# standard YOLO augments -- same as ablation / real baseline. NO albumentations override.
AUGMENTATION_PARAMS = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5, fliplr=0.5,
    mosaic=1.0, degrees=0.0, shear=0.0, perspective=0.0, flipud=0.0,
    mixup=0.0, copy_paste=0.0,
)


def run_name_for(model_size):
    base = f"real_idd_yolo11{model_size}_adamw_100ep"
    return base if _RUN_SEED == 42 else f"{base}_seed{_RUN_SEED}"


def is_complete(model_size):
    return (RUNS_DIR / run_name_for(model_size) / "weights" / "best.pt").exists()


def train_one(model_size, dry_run=False):
    rname   = run_name_for(model_size)
    weights = f"yolo11{model_size}.pt"
    dropout = 0.1 if model_size == "l" else 0.05   # matches train_ablation.py

    if is_complete(model_size):
        print(f"  [SKIP] {rname} already complete (best.pt found)")
        return True

    print(f"\n{'='*64}")
    print(f"  REAL deployment run : {rname}")
    print(f"  weights={weights}  dropout={dropout}  data={DATA_YAML}")
    print(f"{'='*64}")

    if dry_run:
        print("  [DRY RUN] would train here")
        return True

    t0    = time.time()
    model = YOLO(weights)
    model.train(
        data=str(DATA_YAML), project=str(RUNS_DIR), name=rname,
        dropout=dropout, **TRAIN_CFG, **AUGMENTATION_PARAMS,
    )
    print(f"  [DONE] {rname} -- {(time.time()-t0)/60:.1f} min")
    return (RUNS_DIR / rname / "weights" / "best.pt").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["n", "s"], choices=["n", "s", "l"])
    ap.add_argument("--cache", default=TRAIN_CFG["cache"], help="ram (default) | disk | False")
    ap.add_argument("--workers", type=int, default=TRAIN_CFG["workers"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    TRAIN_CFG["cache"]   = False if str(args.cache) == "False" else args.cache
    TRAIN_CFG["workers"] = args.workers

    if not DATA_YAML.exists():
        print(f"[ERROR] data yaml not found: {DATA_YAML}"); return
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  REAL DEPLOYMENT ABLATION -- sizes {args.models}")
    for m in args.models:
        train_one(m, dry_run=args.dry_run)
    print("\n  DONE\n")


if __name__ == "__main__":
    main()
