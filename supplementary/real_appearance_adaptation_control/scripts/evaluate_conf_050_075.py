#!/usr/bin/env python3
"""Evaluate Real + Adapt at manuscript confidence thresholds 0.50 and 0.75."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[3]
CONTROL_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = CONTROL_ROOT / "weights" / "best.pt"
sys.path.insert(0, str(REPO))

from evaluation import eval_ablation_suland as evaluator  # noqa: E402


def main() -> int:
    if not WEIGHTS.is_file():
        raise FileNotFoundError(WEIGHTS)

    evaluator.BATCH_SIZE = 64
    evaluator.DEVICE = "0"
    iid_images = evaluator.get_image_files(evaluator.IDD_IMAGES)
    ood_images = evaluator.get_image_files(evaluator.ODD_IMAGES)
    iid_gt = evaluator.preload_gt_cache(
        iid_images, evaluator.IDD_IMAGES, evaluator.IDD_LABELS
    )
    ood_gt = evaluator.preload_gt_cache(
        ood_images,
        evaluator.ODD_IMAGES,
        evaluator.ODD_LABELS,
        evaluator.ODD_CLASS_REMAP,
    )

    model = YOLO(str(WEIGHTS))
    results = {}
    for confidence in (0.50, 0.75):
        key = f"{confidence:.2f}"
        iid_matrix = evaluator.run_inference(
            model, iid_images, iid_gt, confidence, f"IID-test c={confidence:.2f}"
        )
        ood_matrix = evaluator.run_inference(
            model, ood_images, ood_gt, confidence, f"OOD-val c={confidence:.2f}"
        )
        results[key] = {
            "IID": {
                "n_images": len(iid_images),
                "confusion_matrix": iid_matrix.tolist(),
                "metrics": evaluator.matrix_metrics(iid_matrix),
            },
            "OOD": {
                "n_images": len(ood_images),
                "confusion_matrix": ood_matrix.tolist(),
                "metrics": evaluator.matrix_metrics(ood_matrix),
            },
        }

    payload = {
        "model": "Real + appearance adaptation",
        "weights": str(WEIGHTS.resolve()),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image_size": evaluator.IMGSZ,
        "iou_threshold": evaluator.IOU_THRESHOLD,
        "class_names": evaluator.CLASS_NAMES,
        "ood_ground_truth_class_remap": evaluator.ODD_CLASS_REMAP,
        "results": results,
    }
    json_path = CONTROL_ROOT / "confidence_050_075_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "SSRR 2026 — Real + Appearance Adaptation",
        "Fixed-confidence manuscript evaluator: confidence 0.50 and 0.75",
        "=" * 72,
        "",
        f"Weights: {WEIGHTS}",
        f"Image size: {evaluator.IMGSZ}",
        f"Matching IoU threshold: {evaluator.IOU_THRESHOLD}",
        "OOD ground-truth class remap: 0 <-> 1",
        "",
    ]
    for confidence in ("0.50", "0.75"):
        lines.append(f"CONFIDENCE {confidence}")
        lines.append("-" * 32)
        for domain in ("IID", "OOD"):
            entry = results[confidence][domain]
            metrics = entry["metrics"]
            lines.append(
                f"{domain}: macro-F1={metrics['macro_f1']:.6f}, "
                f"macro-precision={metrics['macro_precision']:.6f}, "
                f"macro-recall={metrics['macro_recall']:.6f}, "
                f"n_images={entry['n_images']}"
            )
            for class_name in evaluator.CLASS_NAMES:
                cls = metrics["classes"][class_name]
                lines.append(
                    f"  {class_name}: F1={cls['f1']:.6f}, "
                    f"precision={cls['precision']:.6f}, recall={cls['recall']:.6f}, "
                    f"TP={cls['tp']}, FP={cls['fp']}, FN={cls['fn']}, "
                    f"support={cls['support']}"
                )
            lines.append(f"  confusion_matrix={entry['confusion_matrix']}")
        lines.append("")

    txt_path = CONTROL_ROOT / "REAL_ADAPT_CONF_050_075_RESULTS.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(txt_path)
    print(json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
