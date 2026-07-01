#!/usr/bin/env python3
"""
SSRR 2026 - Real-data baseline training.

Trains YOLOv11l on the SULAND IDD dataset using the exact same protocol as
train_ablation.py so the real-data row in the ablation table is directly comparable.

  train split : $SSRR_DATA/iid/ITA.yolo/train/images  (22,756 images)
  val   split : $SSRR_DATA/iid/ITA.yolo/val/images    ( 2,836 images)
  test  split : not used during training - evaluated separately via eval_suland.py

Classes match our synthetic model:
  0 = pfm1 (butterfly / PFM-1)
  1 = starfish (PMA-2)

Differences from real_train_example.py (old ICRA script):
  - cos_lr=True   (same as ablation, was False in old script)
  - patience=25   (same as ablation, was 50 in old script)
  - No Albumentations override (real data needs no domain-gap aug)
  - yolo11l only

Usage:
    source configs/env.sh
    python training/train_real_baseline.py
    python train_real_baseline.py --dry-run
"""

import os
import time
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent
RUNS_DIR    = Path(os.environ.get("SSRR_RUNS_BASE",str(Path(__file__).resolve().parents[1]))) / "runs" / "detect"
DATA_YAML   = Path(os.environ.get("SSRR_REAL_YAML", os.environ.get("SSRR_DATA",str(Path(__file__).resolve().parents[1] / "SULAND")) + "/iid/ITA.yolo/ITA_train_val_2.yaml"))

_RUN_SEED   = int(os.environ.get("SSRR_RUN_SEED", 42))   # override for multi-seed runs
RUN_NAME    = "real_idd_yolo11l_adamw_100ep" if _RUN_SEED == 42 else f"real_idd_yolo11l_adamw_100ep_seed{_RUN_SEED}"
WEIGHTS     = "yolo11l.pt"

# ---------------------------------------------------------------------------
# Hyperparameters - identical to train_ablation.py
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
    dropout      = 0.1,
    seed         = _RUN_SEED,
    device       = "0",
    workers      = 12,
    cache        = True,
    exist_ok     = True,
)

# YOLO standard augmentations - same as ablation and SULAND paper protocol
AUGMENTATION_PARAMS = dict(
    hsv_h       = 0.015,
    hsv_s       = 0.7,
    hsv_v       = 0.4,
    translate   = 0.1,
    scale       = 0.5,
    fliplr      = 0.5,
    mosaic      = 1.0,
    degrees     = 0.0,
    shear       = 0.0,
    perspective = 0.0,
    flipud      = 0.0,
    mixup       = 0.0,
    copy_paste  = 0.0,
)


def is_complete():
    return (RUNS_DIR / RUN_NAME / "weights" / "best.pt").exists()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device",  default=TRAIN_CFG["device"])
    parser.add_argument("--batch",   type=int, default=TRAIN_CFG["batch"])
    args = parser.parse_args()

    if not DATA_YAML.exists():
        print(f"[ERROR] Data YAML not found: {DATA_YAML}")
        return

    print(f"\n{'='*64}")
    print(f"  REAL-DATA BASELINE - {RUN_NAME}")
    print(f"  data  : {DATA_YAML}")
    print(f"  runs  : {RUNS_DIR / RUN_NAME}")
    print(f"{'='*64}")

    if is_complete():
        print(f"\n[SKIP] {RUN_NAME} already complete (best.pt found).")
        return

    if args.dry_run:
        print("\n[DRY RUN] Would run:")
        cfg = {**TRAIN_CFG, **AUGMENTATION_PARAMS, "data": str(DATA_YAML),
               "project": str(RUNS_DIR), "name": RUN_NAME, "device": args.device, "batch": args.batch}
        for k, v in cfg.items():
            print(f"  {k:20s} = {v}")
        return

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    model = YOLO(WEIGHTS)
    model.train(
        data    = str(DATA_YAML),
        project = str(RUNS_DIR),
        name    = RUN_NAME,
        device  = args.device,
        batch   = args.batch,
        **{k: v for k, v in TRAIN_CFG.items() if k not in ("device", "batch")},
        **AUGMENTATION_PARAMS,
    )
    elapsed = (time.time() - t0) / 60
    print(f"\n[DONE] {RUN_NAME} - {elapsed:.1f} min")
    print(f"       best.pt → {RUNS_DIR / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
