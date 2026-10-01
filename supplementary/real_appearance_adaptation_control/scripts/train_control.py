#!/usr/bin/env python3
"""Train the real-data YOLO11l fairness control with appearance adaptation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


CONTROL_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training import train_ablation as appearance  # installs the exact synthetic hook
from training import train_real_baseline as real


RUN_NAME = "real_idd_yolo11l_adamw_100ep_with_appearance_adaptation"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", choices=["ram", "disk", "False"], default="ram")
    args = parser.parse_args()

    appearance.set_run_seed(42)
    project = CONTROL_ROOT / "training_run"
    cache = False if args.cache == "False" else args.cache
    config = {
        **real.TRAIN_CFG,
        **real.AUGMENTATION_PARAMS,
        "data": str(real.DATA_YAML),
        "model": str(REPO / "yolo11l.pt"),
        "project": str(project),
        "name": RUN_NAME,
        "device": args.device,
        "workers": args.workers,
        "cache": cache,
        "appearance_adaptation": True,
        "appearance_transform_class": "training.train_ablation.SyntheticToRealAlbumentations",
    }
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "real_appearance_adaptation_fairness_control",
        "single_changed_factor": "synthetic appearance-adaptation stage enabled for real training",
        "training_arguments": config,
    }
    (CONTROL_ROOT / "training_config").mkdir(parents=True, exist_ok=True)
    (CONTROL_ROOT / "training_config" / "config.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    from ultralytics import YOLO

    model = YOLO(str(REPO / "yolo11l.pt"))
    model.train(
        data=str(real.DATA_YAML),
        project=str(project),
        name=RUN_NAME,
        device=args.device,
        workers=args.workers,
        cache=cache,
        batch=real.TRAIN_CFG["batch"],
        **{
            key: value
            for key, value in real.TRAIN_CFG.items()
            if key not in ("device", "workers", "cache", "batch")
        },
        **real.AUGMENTATION_PARAMS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
