#!/usr/bin/env python3
"""Evaluate the SSRR 2026 YOLO11l/30k paper models for AP50 and mAP50-95.

The manuscript's custom fixed-confidence metric evaluator remaps the USA/OOD
ground-truth class IDs because SULAND USA uses the reverse class order.  This
script materializes an equivalent remapped OOD label tree, then runs the
standard Ultralytics detection validator so AP is computed at IoU 0.50 and
over IoU 0.50:0.95.

Outputs are resumable.  Each model/domain directory contains Ultralytics raw
validation plots plus metrics.json; combined_metrics.csv/json summarize all
runs.  Existing successful metrics.json files are reused unless --force is
given.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import torch
import ultralytics
from ultralytics import YOLO


CONFIGS = [
    "real",
    "baseline",
    "h_off",
    "sun_off",
    "t_10",
    "t_5",
    "v_low",
    "v_mid",
    "inv_off",
    "pen_off",
    "s_off",
    "i_1",
    "n_0",
    "n_30",
    "yaw_off",
]

SYNTHETIC_RUN = "{config}_30k_yolo11l_adamw_100ep"
REAL_RUN = "real_idd_yolo11l_adamw_100ep"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Released SSRR repository containing runs/detect.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(os.environ.get("SSRR_DATA", Path(__file__).resolve().parents[1] / "SULAND")),
        help="SULAND root containing iid/ITA.yolo and ood/USA.yolo.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "eval_results"
        / "standard_ap"
        / "yolo11l_30k",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/tmp/ssrr2026_map_eval_data"),
        help="Scratch location for dataset YAMLs and remapped OOD labels.",
    )
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=CONFIGS)
    parser.add_argument("--domains", nargs="+", choices=["IID", "OOD"], default=["IID", "OOD"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def image_files(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def prepare_iid_yaml(data_root: Path, work_dir: Path) -> tuple[Path, int]:
    images = data_root / "iid" / "ITA.yolo" / "test" / "images"
    labels = data_root / "iid" / "ITA.yolo" / "test" / "labels"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f"Incomplete IID split below {data_root}")
    count = len(image_files(images))
    yaml_path = work_dir / "iid_test.yaml"
    write_text(
        yaml_path,
        "\n".join(
            [
                f"path: {data_root / 'iid' / 'ITA.yolo'}",
                "train: test/images",
                "val: test/images",
                "test: test/images",
                "names:",
                "  0: pfm1",
                "  1: starfish",
                "",
            ]
        ),
    )
    return yaml_path, count


def remap_label(src: Path, dst: Path) -> None:
    rows = []
    if src.exists():
        for line in src.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            cls = int(float(fields[0]))
            if cls not in (0, 1):
                raise ValueError(f"Unexpected OOD class {cls} in {src}")
            fields[0] = str(1 - cls)
            rows.append(" ".join(fields))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(("\n".join(rows) + "\n") if rows else "", encoding="utf-8")


def prepare_ood_yaml(data_root: Path, work_dir: Path) -> tuple[Path, int]:
    src_images = data_root / "ood" / "USA.yolo" / "val" / "images"
    src_labels = data_root / "ood" / "USA.yolo" / "val" / "labels"
    if not src_images.is_dir() or not src_labels.is_dir():
        raise FileNotFoundError(f"Incomplete OOD split below {data_root}")

    dst_root = work_dir / "ood_remapped"
    dst_images = dst_root / "images"
    dst_labels = dst_root / "labels"
    images = image_files(src_images)
    for src_img in images:
        rel = src_img.relative_to(src_images)
        dst_img = dst_images / rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            dst_img.symlink_to(src_img)
        remap_label(src_labels / rel.with_suffix(".txt"), dst_labels / rel.with_suffix(".txt"))

    yaml_path = work_dir / "ood_val_remapped.yaml"
    write_text(
        yaml_path,
        "\n".join(
            [
                f"path: {dst_root}",
                "train: images",
                "val: images",
                "test: images",
                "names:",
                "  0: pfm1",
                "  1: starfish",
                "",
            ]
        ),
    )
    return yaml_path, len(images)


def weight_path(repo: Path, config: str) -> Path:
    if config == "real":
        path = repo / "runs" / "detect" / "real_ablation" / REAL_RUN / "weights" / "best.pt"
    else:
        path = (
            repo
            / "runs"
            / "detect"
            / "synthetic_ablation"
            / SYNTHETIC_RUN.format(config=config)
            / "weights"
            / "best.pt"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint for {config}: {path}")
    return path


def environment_record(args: argparse.Namespace, datasets: dict[str, tuple[Path, int]]) -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu,
        "ultralytics": ultralytics.__version__,
        "repo": str(args.repo.resolve()),
        "data": str(args.data.resolve()),
        "image_size": 640,
        "batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "domains": {
            domain: {"yaml": str(yaml.resolve()), "n_images": n_images}
            for domain, (yaml, n_images) in datasets.items()
        },
        "ood_class_remap": {"source_0": 1, "source_1": 0},
        "metric_definition": {
            "AP50": "Ultralytics box mAP at IoU=0.50",
            "mAP50_95": "Ultralytics box mAP averaged over IoU=0.50:0.95",
        },
    }


def metric_payload(config: str, domain: str, weights: Path, n_images: int, metrics, elapsed: float) -> dict:
    box = metrics.box
    class_names = metrics.names
    per_class = []
    for idx, cls_idx in enumerate(box.ap_class_index.tolist()):
        values = box.class_result(idx)
        per_class.append(
            {
                "class_id": int(cls_idx),
                "class_name": str(class_names[int(cls_idx)]),
                "precision": float(values[0]),
                "recall": float(values[1]),
                "AP50": float(values[2]),
                "mAP50_95": float(values[3]),
            }
        )
    return {
        "status": "complete",
        "config": config,
        "domain": domain,
        "weights": str(weights.resolve()),
        "n_images": n_images,
        "image_size": 640,
        "AP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "fitness": float(metrics.fitness),
        "per_class": per_class,
        "results_dict": {k: float(v) for k, v in metrics.results_dict.items()},
        "speed_ms_per_image": {k: float(v) for k, v in metrics.speed.items()},
        "elapsed_seconds": elapsed,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }


def load_completed(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if payload.get("status") == "complete" else None


def export_combined(out_dir: Path, rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda row: (CONFIGS.index(row["config"]), row["domain"]))
    fields = [
        "config",
        "domain",
        "n_images",
        "AP50",
        "mAP50_95",
        "precision",
        "recall",
        "elapsed_seconds",
        "weights",
    ]
    with (out_dir / "combined_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "combined_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    args.repo = args.repo.resolve()
    args.data = args.data.resolve()
    args.out_dir = args.out_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "IID": prepare_iid_yaml(args.data, args.work_dir),
        "OOD": prepare_ood_yaml(args.data, args.work_dir),
    }
    for domain, (_, count) in datasets.items():
        expected = 3743 if domain == "IID" else 4436
        if count != expected:
            raise RuntimeError(f"{domain} image count is {count}, expected {expected}")

    weights = {config: weight_path(args.repo, config) for config in args.configs}
    (args.out_dir / "environment.json").write_text(
        json.dumps(environment_record(args, datasets), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(datasets["IID"][0], args.out_dir / "iid_test.yaml")
    shutil.copy2(datasets["OOD"][0], args.out_dir / "ood_val_remapped.yaml")

    print(f"Ultralytics {ultralytics.__version__}; torch {torch.__version__}; CUDA={torch.cuda.is_available()}")
    print(f"Output: {args.out_dir}")
    print(f"Plan: {len(args.configs)} configs x {len(args.domains)} domains")
    if args.dry_run:
        for config in args.configs:
            print(config, weights[config])
        return 0

    combined = []
    for config in args.configs:
        for domain in args.domains:
            run_dir = args.out_dir / domain / config
            metrics_path = run_dir / "metrics.json"
            prior = None if args.force else load_completed(metrics_path)
            if prior is not None:
                print(f"[reuse] {config:10s} {domain}: AP50={prior['AP50']:.6f} mAP50-95={prior['mAP50_95']:.6f}")
                combined.append(prior)
                continue

            run_dir.mkdir(parents=True, exist_ok=True)
            print(f"[run] {config:10s} {domain} -> {run_dir}", flush=True)
            model = YOLO(str(weights[config]))
            started = time.time()
            metrics = model.val(
                data=str(datasets[domain][0]),
                split="val",
                imgsz=640,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                half=torch.cuda.is_available(),
                plots=True,
                save_json=True,
                project=str(args.out_dir / domain),
                name=config,
                exist_ok=True,
                verbose=False,
            )
            elapsed = time.time() - started
            payload = metric_payload(
                config, domain, weights[config], datasets[domain][1], metrics, elapsed
            )
            metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            combined.append(payload)
            export_combined(args.out_dir, combined)
            print(
                f"[done] {config:10s} {domain}: AP50={payload['AP50']:.6f} "
                f"mAP50-95={payload['mAP50_95']:.6f} ({elapsed:.1f}s)",
                flush=True,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    export_combined(args.out_dir, combined)
    print(f"Complete: {args.out_dir / 'combined_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
