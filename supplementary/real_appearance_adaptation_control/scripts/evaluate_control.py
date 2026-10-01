#!/usr/bin/env python3
"""Evaluate the real-data appearance-adaptation fairness control."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
from ultralytics import YOLO


RELEASED_REPO = Path(__file__).resolve().parents[3]
CONTROL_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = CONTROL_ROOT / "weights" / "best.pt"
WORK_DIR = CONTROL_ROOT / "_eval_work"

sys.path.insert(0, str(RELEASED_REPO))
from evaluation import eval_ablation_suland as fixed  # noqa: E402
from evaluation import eval_standard_ap as standard  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    data_root = Path(os.environ["SSRR_DATA"]).resolve()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    datasets = {
        "IID": standard.prepare_iid_yaml(data_root, WORK_DIR),
        "OOD": standard.prepare_ood_yaml(data_root, WORK_DIR),
    }
    expected = {"IID": 3743, "OOD": 4436}
    for domain, (_, count) in datasets.items():
        if count != expected[domain]:
            raise RuntimeError(f"{domain} image count is {count}, expected {expected[domain]}")

    # Standard detector metrics and all plots emitted by Ultralytics.
    ap_payloads = {}
    for domain, (yaml_path, n_images) in datasets.items():
        out_dir = CONTROL_ROOT / f"{domain}_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(yaml_path, out_dir / yaml_path.name)
        model = YOLO(str(WEIGHTS))
        started = time.time()
        metrics = model.val(
            data=str(yaml_path),
            split="val",
            imgsz=640,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            half=torch.cuda.is_available(),
            plots=True,
            save_json=True,
            project=str(CONTROL_ROOT),
            name=f"{domain}_eval",
            exist_ok=True,
            verbose=False,
        )
        payload = standard.metric_payload(
            "real_yolo11l_with_appearance_adaptation",
            domain,
            WEIGHTS,
            n_images,
            metrics,
            time.time() - started,
        )
        write_json(out_dir / "standard_metrics.json", payload)
        ap_payloads[domain] = payload
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Current-manuscript operating point: macro-F1 at confidence 0.25,
    # retaining its deliberately permissive IoU >= 0.001 matching rule.
    fixed.BATCH_SIZE = args.batch
    fixed.DEVICE = args.device
    iid_images = fixed.get_image_files(fixed.IDD_IMAGES)
    odd_images = fixed.get_image_files(fixed.ODD_IMAGES)
    iid_gt = fixed.preload_gt_cache(iid_images, fixed.IDD_IMAGES, fixed.IDD_LABELS)
    odd_gt = fixed.preload_gt_cache(
        odd_images, fixed.ODD_IMAGES, fixed.ODD_LABELS, fixed.ODD_CLASS_REMAP
    )
    model = YOLO(str(WEIGHTS))
    matrices = {
        "IID": fixed.run_inference(model, iid_images, iid_gt, 0.25, "IID-test c=0.25"),
        "OOD": fixed.run_inference(model, odd_images, odd_gt, 0.25, "OOD-val c=0.25"),
    }
    fixed_payloads = {}
    for domain, matrix in matrices.items():
        payload = {
            "domain": domain,
            "weights": str(WEIGHTS.resolve()),
            "confidence": 0.25,
            "iou_threshold": fixed.IOU_THRESHOLD,
            "class_names": fixed.CLASS_NAMES,
            "ood_ground_truth_class_remap": fixed.ODD_CLASS_REMAP if domain == "OOD" else None,
            "confusion_matrix": matrix.tolist(),
            "metrics": fixed.matrix_metrics(matrix),
        }
        write_json(CONTROL_ROOT / f"{domain}_eval" / "operating_point_metrics.json", payload)
        fixed_payloads[domain] = payload

    summary = {
        domain: {
            "AP50": ap_payloads[domain]["AP50"],
            "mAP50_95": ap_payloads[domain]["mAP50_95"],
            "macro_F1_at_conf_0.25": fixed_payloads[domain]["metrics"]["macro_f1"],
        }
        for domain in ("IID", "OOD")
    }
    write_json(CONTROL_ROOT / "metrics_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
