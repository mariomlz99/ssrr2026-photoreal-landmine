#!/usr/bin/env python3
"""
SSRR 2026 - Ablation study training runner.

Trains YOLO11 on all 42 ablation datasets (14 configs × 3 sizes).
Generates data.yaml files automatically on first run.
Skips runs that already have a completed best.pt (safe to re-run after interruption).

Default priority order: l → s → n  (run L first for all configs)

Usage (from SSRR_EXPERIMENTS/):
    python train_ablation.py                          # L only, all 42 configs
    python train_ablation.py --models l s             # L then S
    python train_ablation.py --models l s n         # all 3 sizes
    python train_ablation.py --only baseline v_low    # specific configs only
    python train_ablation.py --sizes 30k              # only 30k datasets
    python train_ablation.py --dry-run                # print plan, no training
    python train_ablation.py --list                   # list all planned runs
"""

import argparse
import csv
import gc
import os
import random
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import numpy as np
import torch
import ultralytics.data.augment as _augment
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parent
DATASETS    = Path(os.environ.get("SSRR_DATASETS",str(Path(__file__).resolve().parents[1] / "datasets")))
RUNS_DIR    = Path(os.environ.get("SSRR_RUNS_BASE",str(Path(__file__).resolve().parents[1]))) / "runs" / "detect" / "synthetic_ablation"
LOG_CSV     = ROOT / "training_log.csv"

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
ALL_CONFIGS = [
    "baseline",
    "v_low", "v_mid",
    "h_off", "s_off",
    "i_1",
    "t_10", "t_5",
    "n_0", "n_30",
    "inv_off",
    "sun_off", "yaw_off", "pen_off",
]
ALL_SIZES   = ["10k", "20k", "30k"]
ALL_MODELS  = ["l", "s", "n"]     # priority order

CLASS_NAMES = ["pfm1", "starfish"]

# ---------------------------------------------------------------------------
# AdamW config (identical to old training script)
# ---------------------------------------------------------------------------
OPTIMIZER_CFG = {"name": "AdamW", "lr0": 3e-4}

SEED = 42
# Run seed = single source of truth for EVERY stochastic component: model-head init,
# dropout, the YOLO augmentation decisions, AND the Albumentations recipe below
# (which was previously unseeded -> fresh OS entropy on every launch). Override per
# run for multi-seed error-bar experiments:  SSRR_RUN_SEED=43 python train_ablation.py
_RUN_SEED = int(os.environ.get("SSRR_RUN_SEED", SEED))
random.seed(_RUN_SEED)
np.random.seed(_RUN_SEED)


def set_run_seed(seed):
    """Set the run seed in-process. Call before each .train() when sweeping seeds so
    the next-built Albumentations recipe (and run name) tracks the new seed."""
    global _RUN_SEED
    _RUN_SEED = int(seed)
    random.seed(_RUN_SEED)
    np.random.seed(_RUN_SEED)

# ---------------------------------------------------------------------------
# Albumentations domain-gap augmentation  (same as old script)
# ---------------------------------------------------------------------------
class SyntheticToRealAlbumentations:
    def __init__(self, p=1.0, *args, **kwargs):
        self.p = p
        self.transform = A.Compose([
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.05), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=(7, 15), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            ], p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.4),
            A.ImageCompression(quality_range=(50, 85), p=0.3),
        ], seed=_RUN_SEED)   # seed the recipe (Albumentations 2.x): reproducible,
        # and tracks the run seed so multi-seed replicates are genuinely independent.

    def __call__(self, labels):
        if random.random() > self.p:
            return labels
        img = labels.get("img")
        if img is not None and isinstance(img, np.ndarray):
            try:
                labels["img"] = self.transform(image=img)["image"]
            except Exception:
                pass
        return labels


_augment.Albumentations = SyntheticToRealAlbumentations

# ---------------------------------------------------------------------------
# YOLO standard augmentation params (same as SULAND paper / old script)
# ---------------------------------------------------------------------------
AUGMENTATION_PARAMS = {
    "hsv_h":       0.015,
    "hsv_s":       0.7,
    "hsv_v":       0.4,
    "translate":   0.1,
    "scale":       0.5,
    "fliplr":      0.5,
    "mosaic":      1.0,
    "degrees":     0.0,
    "shear":       0.0,
    "perspective": 0.0,
    "flipud":      0.0,
    "mixup":       0.0,
    "copy_paste":  0.0,
}

# ---------------------------------------------------------------------------
# data.yaml generation
# ---------------------------------------------------------------------------
def generate_data_yaml(dataset_dir: Path) -> Path:
    yaml_path = dataset_dir / "data.yaml"
    if yaml_path.exists():
        return yaml_path
    content = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    yaml_path.write_text(content)
    return yaml_path


def generate_all_data_yamls():
    count = 0
    for ds in sorted(DATASETS.iterdir()):
        if ds.is_dir() and (ds / "train" / "images").exists():
            generate_data_yaml(ds)
            count += 1
    print(f"[YAML] Generated/verified data.yaml for {count} datasets.")

# ---------------------------------------------------------------------------
# Run queue
# ---------------------------------------------------------------------------
def build_run_queue(models, configs, sizes):
    """Returns ordered list of (config, size, model_size) tuples.
    Priority: model first (l→m→s→n), then config, then size (30k→20k→10k).
    """
    queue = []
    for model in models:
        for config in configs:
            for size in sorted(sizes, key=lambda s: int(s[:-1]), reverse=True):
                ds = DATASETS / f"{config}_{size}"
                if not ds.exists():
                    continue
                queue.append((config, size, model))
    return queue


def run_name_for(config, size, model_size):
    base = f"{config}_{size}_yolo11{model_size}_adamw_100ep"
    # Canonical seed (42) keeps the original name so existing eval/paths resolve;
    # alternate seeds get a _seed{N} suffix so replicates don't collide.
    return base if _RUN_SEED == SEED else f"{base}_seed{_RUN_SEED}"


def is_complete(config, size, model_size):
    rname = run_name_for(config, size, model_size)
    return (RUNS_DIR / rname / "weights" / "best.pt").exists()

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_training(config, size, model_size, args):
    ds_dir   = DATASETS / f"{config}_{size}"
    yaml_path = generate_data_yaml(ds_dir)
    rname     = run_name_for(config, size, model_size)
    weights   = f"yolo11{model_size}.pt"

    dropout   = 0.1 if model_size == "l" else 0.05
    cache_val = False if args.cache == "False" else args.cache

    print(f"\n{'#' * 64}")
    print(f"  RUN : {rname}")
    print(f"  DATA: {yaml_path}")
    print(f"  GPU : {args.device}  |  batch={args.batch}  |  workers={args.workers}")
    print(f"{'#' * 64}\n")

    model = YOLO(weights)
    try:
        model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            patience=args.patience,
            imgsz=args.imgsz,
            batch=args.batch,
            optimizer=OPTIMIZER_CFG["name"],
            lr0=OPTIMIZER_CFG["lr0"],
            lrf=0.01,
            cos_lr=True,
            weight_decay=0.0005,
            dropout=dropout,
            seed=_RUN_SEED,
            project=str(RUNS_DIR),
            name=rname,
            device=args.device,
            workers=args.workers,
            cache=cache_val,
            exist_ok=True,
            **AUGMENTATION_PARAMS,
        )
    finally:
        # Release GPU memory and PyTorch allocator pool before next run.
        # Without this, decoded image caches and CUDA tensors from DataLoader
        # workers accumulate across runs and exhaust RAM (~36 GB per 30k run).
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def log_run(config, size, model_size, status, elapsed_min):
    fieldnames = ["timestamp", "config", "size", "model", "run_name", "status", "elapsed_min"]
    write_header = not LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow({
            "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M"),
            "config":      config,
            "size":        size,
            "model":       f"yolo11{model_size}",
            "run_name":    run_name_for(config, size, model_size),
            "status":      status,
            "elapsed_min": f"{elapsed_min:.1f}",
        })

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SSRR ablation training runner")
    parser.add_argument("--models",   nargs="+", default=["l"],
                        choices=ALL_MODELS,
                        help="Model sizes to train, in priority order (default: l)")
    parser.add_argument("--only",     nargs="+", default=None,
                        help="Train only these config names (e.g. baseline v_low)")
    parser.add_argument("--sizes",    nargs="+", default=ALL_SIZES,
                        choices=ALL_SIZES,
                        help="Dataset sizes to include (default: 10k 20k 30k)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print the full run plan without executing")
    parser.add_argument("--list",     action="store_true",
                        help="List all runs with their completion status and exit")
    parser.add_argument("--device",   default="0",
                        help="CUDA device index (default: 0)")
    parser.add_argument("--epochs",   type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--imgsz",    type=int, default=640)
    parser.add_argument("--batch",    type=int, default=32)
    parser.add_argument("--workers",  type=int, default=4,
                        help="DataLoader workers per run (default: 4; 12 causes RAM exhaustion)")
    parser.add_argument("--sleep",    type=float, default=30.0,
                        help="Cooldown seconds between runs (default: 30)")
    parser.add_argument("--cache",    default="disk",
                        choices=["disk", "ram", "False"],
                        help="Image cache mode: disk (default, safe), ram (fast but 36 GB/run), False (none)")
    return parser.parse_args()


def main():
    args   = parse_args()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    configs = args.only if args.only else ALL_CONFIGS
    unknown = set(configs) - set(ALL_CONFIGS)
    if unknown:
        print(f"[ERROR] Unknown config(s): {unknown}")
        print(f"        Valid: {ALL_CONFIGS}")
        return

    generate_all_data_yamls()

    queue    = build_run_queue(args.models, configs, args.sizes)
    pending  = [(c, s, m) for c, s, m in queue if not is_complete(c, s, m)]
    skipped  = [(c, s, m) for c, s, m in queue if     is_complete(c, s, m)]

    print(f"\n{'='*64}")
    print(f"  SSRR ABLATION TRAINING PLAN")
    print(f"  Models   : {args.models}")
    print(f"  Configs  : {len(configs)}")
    print(f"  Sizes    : {args.sizes}")
    print(f"  Total    : {len(queue)} runs  ({len(skipped)} already done, {len(pending)} pending)")
    print(f"{'='*64}")

    if args.list:
        for c, s, m in queue:
            done = "✓ done" if is_complete(c, s, m) else "  todo"
            print(f"  [{done}]  {run_name_for(c, s, m)}")
        return

    if args.dry_run:
        print("\nDRY RUN - pending runs:")
        for i, (c, s, m) in enumerate(pending, 1):
            print(f"  {i:3d}. {run_name_for(c, s, m)}")
        print(f"\nSkipped (already complete): {len(skipped)}")
        return

    if not pending:
        print("\nAll runs already complete.")
        return

    print(f"\nStarting {len(pending)} runs...\n")

    for idx, (config, size, model_size) in enumerate(pending, 1):
        print(f"\n>>> RUN {idx}/{len(pending)}")
        t0 = time.time()
        try:
            run_training(config, size, model_size, args)
            elapsed = (time.time() - t0) / 60
            log_run(config, size, model_size, "OK", elapsed)
            print(f">>> DONE in {elapsed:.1f} min")
        except Exception as e:
            elapsed = (time.time() - t0) / 60
            log_run(config, size, model_size, f"ERROR: {e}", elapsed)
            print(f"[ERROR] {e}")

        if idx < len(pending) and args.sleep > 0:
            # Log system health so overheating / memory issues are visible in the log.
            try:
                import subprocess, shutil
                if shutil.which("nvidia-smi"):
                    smi = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.free,fan.speed",
                         "--format=csv,noheader,nounits"], text=True
                    ).strip()
                    print(f"  GPU  state : temp={smi.split(',')[0].strip()}°C  "
                          f"mem_used={smi.split(',')[1].strip()}MiB  "
                          f"fan={smi.split(',')[3].strip()}%")
                with open("/proc/meminfo") as mf:
                    lines = {l.split(':')[0]: int(l.split(':')[1].strip().split()[0])
                             for l in mf if ':' in l}
                avail_gb = lines.get("MemAvailable", 0) / 1024**2
                print(f"  Host RAM   : {avail_gb:.1f} GB available")
            except Exception:
                pass
            print(f"Cooling down {args.sleep}s...")
            time.sleep(args.sleep)

    print(f"\n{'='*64}")
    print(f"  ALL DONE - log saved to {LOG_CSV}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
