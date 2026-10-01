#!/usr/bin/env python3
"""Resume the real-data appearance-adaptation fairness control."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


CONTROL_ROOT = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[3]
RUN_NAME = "real_idd_yolo11l_adamw_100ep_with_appearance_adaptation"
LAST = CONTROL_ROOT / "training_run" / RUN_NAME / "weights" / "last.pt"
sys.path.insert(0, str(REPO))

from training import train_ablation as appearance  # installs the exact synthetic hook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", choices=["ram", "disk", "False"], default="ram")
    args = parser.parse_args()
    if not LAST.is_file():
        raise FileNotFoundError(LAST)
    appearance.set_run_seed(42)

    from ultralytics import YOLO

    YOLO(str(LAST)).train(
        resume=True,
        device=args.device,
        workers=args.workers,
        cache=False if args.cache == "False" else args.cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
