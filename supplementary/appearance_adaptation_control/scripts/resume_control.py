#!/usr/bin/env python3
"""Resume the interrupted appearance-adaptation control from last.pt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CONTROL_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
RUN_NAME = "baseline_30k_yolo11l_no_appearance_adaptation_seed42"
LAST = CONTROL_ROOT / "training_run" / RUN_NAME / "weights" / "last.pt"
sys.path.insert(0, str(REPO))

from training import train_ablation as baseline  # noqa: E402


class AppearanceAdaptationOff:
    """Drop-in Ultralytics hook that leaves images and labels unchanged."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, labels):
        return labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", choices=["ram", "disk", "False"], default="ram")
    args = parser.parse_args()
    if not LAST.is_file():
        raise FileNotFoundError(LAST)

    baseline._augment.Albumentations = AppearanceAdaptationOff

    from ultralytics import YOLO

    model = YOLO(str(LAST))
    model.train(
        resume=True,
        device=args.device,
        workers=args.workers,
        cache=False if args.cache == "False" else args.cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
