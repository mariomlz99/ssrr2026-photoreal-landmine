#!/usr/bin/env python3
"""Train the single SSRR'26 appearance-adaptation control.

Only the extra synthetic-only Albumentations stage is replaced by a no-op.
All YOLO training arguments are imported from the released Baseline runner.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


CONTROL_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training import train_ablation as baseline  # noqa: E402


class AppearanceAdaptationOff:
    """Drop-in Ultralytics hook that leaves images and labels unchanged."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, labels):
        return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", choices=["ram", "disk", "False"], default="ram")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = baseline.DATASETS / "baseline_30k"
    data_yaml = baseline.generate_data_yaml(dataset)
    project = CONTROL_ROOT / "training_run"
    run_name = "baseline_30k_yolo11l_no_appearance_adaptation_seed42"

    baseline._augment.Albumentations = AppearanceAdaptationOff
    baseline.set_run_seed(42)

    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "appearance_adaptation_control",
        "single_changed_factor": "synthetic appearance-adaptation stage disabled",
        "appearance_adaptation": False,
        "standard_yolo_augmentations": baseline.AUGMENTATION_PARAMS,
        "dataset": str(dataset),
        "data_yaml": str(data_yaml),
        "initial_checkpoint": "yolo11l.pt",
        "model": "YOLO11l",
        "seed": 42,
        "optimizer": "AdamW",
        "lr0": 3e-4,
        "lrf": 0.01,
        "cos_lr": True,
        "weight_decay": 5e-4,
        "dropout": 0.1,
        "batch": 32,
        "epochs": 100,
        "patience": 25,
        "imgsz": 640,
        "device": args.device,
        "workers": args.workers,
        "cache": args.cache,
        "project": str(project),
        "run_name": run_name,
    }
    CONTROL_ROOT.mkdir(parents=True, exist_ok=True)
    (CONTROL_ROOT / "training_config" / "config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from ultralytics import YOLO

    model = YOLO("yolo11l.pt")
    model.train(
        data=str(data_yaml),
        epochs=100,
        patience=25,
        imgsz=640,
        batch=32,
        optimizer="AdamW",
        lr0=3e-4,
        lrf=0.01,
        cos_lr=True,
        weight_decay=5e-4,
        dropout=0.1,
        seed=42,
        project=str(project),
        name=run_name,
        device=args.device,
        workers=args.workers,
        cache=False if args.cache == "False" else args.cache,
        exist_ok=True,
        **baseline.AUGMENTATION_PARAMS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
