#!/usr/bin/env python3
"""
SSRR 2026 - Ablation study: SULAND evaluation.

Evaluates all 14 ablation configs (30k, yolo11l) on:
  IDD : ITA.yolo / test      (held-out; real model tuned best.pt on val, so val
                              would over-evaluate it -- test is fair to all)
  ODD : USA.yolo / val       (USA has no train/test; val IS the OOD test set)

Conf thresholds : 0.25 / 0.50 / 0.75
IoU threshold   : 0.001

ODD class inversion: USA.yolo GT defines 0=starfish 1=butterfly - the opposite
of our model space (0=pfm1 1=starfish). Labels are remapped on load.

Outputs  (eval_results/ablation_suland/<run-timestamp>/):
  per_model/<config>.json      raw confusion matrices + metrics per conf
  summary.csv                  one row per (config, conf, domain)
  report.txt                   full per-class report
  table_macroF1.txt            Macro-F1 grid: all 14 configs x 3 conf thresholds x 2 domains

Usage:
    python eval_ablation_suland.py
    python eval_ablation_suland.py --resume
    python eval_ablation_suland.py --outdir eval_results/ablation_suland/custom_run
"""

import argparse
import csv
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SULAND_ROOT = Path(os.environ.get("SSRR_DATA",str(Path(__file__).resolve().parents[1] / "SULAND")))
SSRR_ROOT   = Path(os.environ.get("SSRR_RUNS_BASE",str(Path(__file__).resolve().parents[1])))
RUNS_DIR    = SSRR_ROOT / "runs" / "detect" / "synthetic_ablation"
_RUN_SEED    = int(os.environ.get("SSRR_RUN_SEED", 42))
_SEED_SUFFIX = "" if _RUN_SEED == 42 else f"_seed{_RUN_SEED}"   # resolve _seed{N} weights for alt seeds

# IDD on the held-out TEST split (fair to the real model, which selected its
# best.pt on ITA-val). ODD stays on USA-val (the only USA split = OOD test).
IDD_SPLIT   = "test"
ODD_SPLIT   = "val"
IDD_IMAGES  = SULAND_ROOT / "iid" / "ITA.yolo" / IDD_SPLIT / "images"
IDD_LABELS  = SULAND_ROOT / "iid" / "ITA.yolo" / IDD_SPLIT / "labels"
ODD_IMAGES  = SULAND_ROOT / "ood" / "USA.yolo" / ODD_SPLIT / "images"
ODD_LABELS  = SULAND_ROOT / "ood" / "USA.yolo" / ODD_SPLIT / "labels"
IDD_TAG     = f"IDD-{IDD_SPLIT}"
ODD_TAG     = f"ODD-{ODD_SPLIT}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES     = ["pfm1", "starfish"]
NC              = len(CLASS_NAMES)
CONF_THRESHOLDS = [0.25, 0.50, 0.75]
IOU_THRESHOLD   = 0.001
DEVICE          = "0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = int(os.environ.get("SSRR_EVAL_BATCH", 256))   # lower (e.g. 64) to fit a busy/smaller GPU
HALF            = torch.cuda.is_available()
IMGSZ           = 640
VALID_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ODD_CLASS_REMAP = {0: 1, 1: 0}

ALL_CONFIGS = [
    "baseline",
    "v_low",  "v_mid",
    "h_off",  "s_off",
    "i_1",
    "t_10",   "t_5",
    "n_0",    "n_30",
    "inv_off",
    "sun_off", "yaw_off", "pen_off",
]
CONFIG_LABELS = {
    "baseline": "Baseline", "v_low":  "V-Low",   "v_mid":  "V-Mid",
    "h_off":   "H-Off",    "s_off":  "S-Off",   "i_1":    "I-1",
    "t_10":    "T-10",     "t_5":    "T-5",      "n_0":    "N-0",
    "n_30":    "N-30",     "inv_off":"Inv-Off",  "sun_off":"Sun-Off",
    "yaw_off": "Yaw-Off",  "pen_off":"Pen-Off",
}

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_image_files(root: Path):
    return sorted(str(p) for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in VALID_EXTS)


def preload_gt_cache(images: list, images_root: Path, labels_root: Path,
                     remap: dict = None) -> dict:
    cache = {}
    for img_path in images:
        rel   = Path(img_path).relative_to(images_root)
        lbl   = Path(labels_root) / rel.with_suffix(".txt")
        boxes = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls = int(float(parts[0]))
                    if remap:
                        cls = remap.get(cls, cls)
                    boxes.append((cls, *map(float, parts[1:5])))
        cache[img_path] = boxes
    return cache


def gt_from_cache(img_path: str, cache: dict, shape: tuple) -> list:
    h, w = shape
    return [(cls, (xc-nw/2)*w, (yc-nh/2)*h, (xc+nw/2)*w, (yc+nh/2)*h)
            for cls, xc, yc, nw, nh in cache.get(img_path, [])]


def update_matrix(matrix: np.ndarray, gt_boxes: list, pred_result) -> None:
    if pred_result.boxes is None or len(pred_result.boxes) == 0:
        for cls, *_ in gt_boxes:
            matrix[cls, NC] += 1
        return
    pboxes = pred_result.boxes.xyxy.cpu().numpy()
    pcls   = pred_result.boxes.cls.cpu().numpy().astype(np.int32)
    P      = len(pcls)
    if not gt_boxes:
        for pi in range(P):
            matrix[NC, pcls[pi]] += 1
        return
    G      = len(gt_boxes)
    gboxes = np.array([[x1,y1,x2,y2] for _,x1,y1,x2,y2 in gt_boxes], dtype=np.float32)
    gcls   = np.array([c for c,*_ in gt_boxes], dtype=np.int32)
    xl     = np.maximum(gboxes[:,0:1], pboxes[:,0])
    yt     = np.maximum(gboxes[:,1:2], pboxes[:,1])
    xr     = np.minimum(gboxes[:,2:3], pboxes[:,2])
    yb     = np.minimum(gboxes[:,3:4], pboxes[:,3])
    inter  = np.maximum(0.0, xr-xl) * np.maximum(0.0, yb-yt)
    ag     = (gboxes[:,2]-gboxes[:,0]) * (gboxes[:,3]-gboxes[:,1])
    ap     = (pboxes[:,2]-pboxes[:,0]) * (pboxes[:,3]-pboxes[:,1])
    union  = ag[:,None] + ap[None,:] - inter
    iou_mat = np.where(union > 0.0, inter/union, 0.0)
    used   = np.zeros(P, dtype=bool)
    for g in range(G):
        row = iou_mat[g].copy(); row[used] = -1.0
        bi  = int(np.argmax(row))
        if row[bi] >= IOU_THRESHOLD:
            matrix[gcls[g], pcls[bi]] += 1
            used[bi] = True
        else:
            matrix[gcls[g], NC] += 1
    for pi in range(P):
        if not used[pi]:
            matrix[NC, pcls[pi]] += 1


def matrix_metrics(matrix: np.ndarray) -> dict:
    classes = {}
    f1s, precs, recs = [], [], []
    for i, name in enumerate(CLASS_NAMES):
        tp      = matrix[i, i]
        fp      = matrix[:NC, i].sum() - tp + matrix[NC, i]
        fn      = matrix[i, :NC].sum() - tp + matrix[i, NC]
        support = matrix[i, :].sum()
        prec    = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        rec     = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1      = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        classes[name] = {
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "support": int(support),
            "precision": round(prec, 6),
            "recall":    round(rec,  6),
            "f1":        round(f1,   6),
        }
        f1s.append(f1); precs.append(prec); recs.append(rec)
    supports = [classes[n]["support"] for n in CLASS_NAMES]
    total_s  = sum(supports) or 1
    return {
        "classes":        classes,
        "macro_f1":       round(float(np.mean(f1s)),   6),
        "macro_precision":round(float(np.mean(precs)), 6),
        "macro_recall":   round(float(np.mean(recs)),  6),
        "weighted_f1":    round(sum(f1s[i]*supports[i] for i in range(NC))/total_s, 6),
        "total_gt":       int(sum(supports)),
    }


def _live_stats(matrix: np.ndarray, tag: str, done: int, total: int) -> None:
    m  = matrix_metrics(matrix)
    p1 = m["classes"]["pfm1"]
    sf = m["classes"]["starfish"]
    print(f"\r  [{tag}] {done:>5}/{total}"
          f"  MacF1={m['macro_f1']:.3f}"
          f"  pfm1 Rec={p1['recall']:.3f} F1={p1['f1']:.3f}"
          f" TP={p1['tp']:>5} FP={p1['fp']:>5} FN={p1['fn']:>5}"
          f"  sfsh Rec={sf['recall']:.3f} F1={sf['f1']:.3f}",
          end="", flush=True)


def run_inference(model, images: list, gt_cache: dict, conf: float, tag: str) -> np.ndarray:
    matrix = np.zeros((NC+1, NC+1), dtype=np.float64)
    total  = len(images)
    print(f"\n  -- {tag} --  ({total} images, conf={conf})")
    for start in range(0, total, BATCH_SIZE):
        batch   = images[start:start+BATCH_SIZE]
        results = model.predict(
            source=batch, conf=conf, batch=len(batch),
            device=DEVICE, half=HALF, imgsz=IMGSZ, verbose=False,
        )
        for img_path, pred in zip(batch, results):
            update_matrix(matrix, gt_from_cache(img_path, gt_cache, pred.orig_shape), pred)
        _live_stats(matrix, tag, min(start+len(batch), total), total)
    print()
    return matrix


# ---------------------------------------------------------------------------
# Evaluate one model
# ---------------------------------------------------------------------------

def evaluate_model(config: str, weight_path: Path,
                   idd_images: list, idd_gt: dict,
                   odd_images: list, odd_gt: dict) -> dict:
    print(f"\n  Loading: {config}", flush=True)
    model  = YOLO(str(weight_path))
    result = {}

    for conf in CONF_THRESHOLDS:
        cs = f"{conf:.2f}"
        print(f"\n  conf={conf}", flush=True)
        t0      = time.time()
        idd_mat = run_inference(model, idd_images, idd_gt, conf, f"{IDD_TAG} c={conf}")
        t1      = time.time()
        odd_mat = run_inference(model, odd_images, odd_gt, conf, f"{ODD_TAG} c={conf}")
        t2      = time.time()

        result[cs] = {
            "idd": {"n_imgs": len(idd_images), "matrix": idd_mat.tolist(),
                    "metrics": matrix_metrics(idd_mat)},
            "odd": {"n_imgs": len(odd_images), "matrix": odd_mat.tolist(),
                    "metrics": matrix_metrics(odd_mat)},
        }
        im = result[cs]["idd"]["metrics"]
        om = result[cs]["odd"]["metrics"]
        print(f"  IDD ({t1-t0:.0f}s) MacF1={im['macro_f1']:.3f}"
              f"  pfm1 F1={im['classes']['pfm1']['f1']:.3f} Rec={im['classes']['pfm1']['recall']:.3f}"
              f"  sfsh F1={im['classes']['starfish']['f1']:.3f} Rec={im['classes']['starfish']['recall']:.3f}", flush=True)
        print(f"  ODD ({t2-t1:.0f}s) MacF1={om['macro_f1']:.3f}"
              f"  pfm1 F1={om['classes']['pfm1']['f1']:.3f} Rec={om['classes']['pfm1']['recall']:.3f}"
              f"  sfsh F1={om['classes']['starfish']['f1']:.3f} Rec={om['classes']['starfish']['recall']:.3f}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    return result


# ---------------------------------------------------------------------------
# Reports & tables
# ---------------------------------------------------------------------------

def make_macroF1_grid(all_results: dict) -> str:
    col_keys   = [(dom, f"{c:.2f}") for dom in ["idd","odd"] for c in CONF_THRESHOLDS]
    col_labels = [f"{'IDD' if d=='idd' else 'ODD'}-{c}" for d, c in col_keys]
    cw    = 14
    cfg_w = 10
    hdr   = f"  {'Config':<{cfg_w}}  " + "  ".join(f"{l:>{cw}}" for l in col_labels)
    sep   = "  " + "-" * len(hdr.strip())

    lines = ["", "  MACRO-F1 GRID  (14 configs x 2 domains x 3 conf thresholds)", sep, hdr, sep]

    best = {k: (None, -1.0) for k in col_keys}
    rows = {}
    for config in ALL_CONFIGS:
        if config not in all_results:
            continue
        rows[config] = {}
        for dom, cs in col_keys:
            mf1 = all_results[config][cs][dom]["metrics"]["macro_f1"]
            rows[config][(dom, cs)] = mf1
            if mf1 > best[(dom, cs)][1]:
                best[(dom, cs)] = (config, mf1)

    for config in ALL_CONFIGS:
        if config not in rows:
            lines.append(f"  {CONFIG_LABELS.get(config, config):<{cfg_w}}  [MISSING]")
            continue
        cells = []
        for k in col_keys:
            v      = rows[config][k]
            marker = "*" if best[k][0] == config else " "
            cells.append(f"{v:.3f}{marker}")
        lines.append(f"  {CONFIG_LABELS.get(config, config):<{cfg_w}}  " +
                     "  ".join(f"{c:>{cw}}" for c in cells))

    lines += [sep, "  * = best in column", ""]
    return "\n".join(lines)


def make_report(all_results: dict) -> str:
    lines = [
        "SSRR 2026 -- SULAND Ablation Evaluation",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"IoU threshold : {IOU_THRESHOLD}",
        f"IDD : ITA.yolo/{IDD_SPLIT}   ODD : USA.yolo/{ODD_SPLIT}",
        "",
    ]
    for config in ALL_CONFIGS:
        if config not in all_results:
            continue
        lines.append(f"\n{'='*72}\n  {CONFIG_LABELS.get(config, config)}\n{'='*72}")
        for conf in CONF_THRESHOLDS:
            cs = f"{conf:.2f}"
            lines.append(f"\n  conf={conf}")
            for dom in ["idd", "odd"]:
                r  = all_results[config][cs][dom]
                m  = r["metrics"]
                lines += [
                    f"  -- {IDD_TAG if dom=='idd' else ODD_TAG} ({r['n_imgs']} images) --",
                    f"  {'Class':<12} {'Support':>8} {'TP':>7} {'FP':>7} {'FN':>7}"
                    f" {'Prec':>8} {'Rec':>8} {'F1':>8}",
                    "  " + "-"*72,
                ]
                for cls in CLASS_NAMES:
                    c = m["classes"][cls]
                    lines.append(
                        f"  {cls:<12} {c['support']:>8} {c['tp']:>7} {c['fp']:>7} {c['fn']:>7}"
                        f" {c['precision']:>8.4f} {c['recall']:>8.4f} {c['f1']:>8.4f}")
                lines += [
                    "  " + "-"*72,
                    f"  {'Macro':>12}  {'':>8}  {'':>7}  {'':>7}  {'':>7}"
                    f" {m['macro_precision']:>8.4f} {m['macro_recall']:>8.4f} {m['macro_f1']:>8.4f}",
                    "",
                ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def export_csv(all_results: dict, out_path: Path) -> None:
    fieldnames = [
        "config", "conf", "domain", "n_imgs",
        "macro_f1", "macro_precision", "macro_recall", "weighted_f1",
        "pfm1_tp", "pfm1_fp", "pfm1_fn", "pfm1_support",
        "pfm1_precision", "pfm1_recall", "pfm1_f1",
        "starfish_tp", "starfish_fp", "starfish_fn", "starfish_support",
        "starfish_precision", "starfish_recall", "starfish_f1",
    ]
    rows = []
    for config in ALL_CONFIGS:
        if config not in all_results:
            continue
        for conf in CONF_THRESHOLDS:
            cs = f"{conf:.2f}"
            for dom in ["idd", "odd"]:
                r  = all_results[config][cs][dom]
                m  = r["metrics"]
                p1 = m["classes"]["pfm1"]
                sf = m["classes"]["starfish"]
                rows.append({
                    "config": config, "conf": conf, "domain": dom, "n_imgs": r["n_imgs"],
                    "macro_f1": m["macro_f1"], "macro_precision": m["macro_precision"],
                    "macro_recall": m["macro_recall"], "weighted_f1": m["weighted_f1"],
                    "pfm1_tp": p1["tp"], "pfm1_fp": p1["fp"], "pfm1_fn": p1["fn"],
                    "pfm1_support": p1["support"], "pfm1_precision": p1["precision"],
                    "pfm1_recall": p1["recall"], "pfm1_f1": p1["f1"],
                    "starfish_tp": sf["tp"], "starfish_fp": sf["fp"], "starfish_fn": sf["fn"],
                    "starfish_support": sf["support"], "starfish_precision": sf["precision"],
                    "starfish_recall": sf["recall"], "starfish_f1": sf["f1"],
                })
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV  -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--size", default="l", choices=["l", "s", "n"],
                        help="YOLO11 model size to evaluate (default: l)")
    parser.add_argument("--datasize", default="30k", choices=["10k", "20k", "30k"],
                        help="Synthetic dataset size the model was trained on (default: 30k)")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Subset of configs to evaluate (default: all 14)")
    args = parser.parse_args()

    configs = args.configs if args.configs else ALL_CONFIGS

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.outdir) if args.outdir else (
           SSRR_ROOT / "eval_results" / "ablation_suland" / f"run_{ts}_yolo11{args.size}_{args.datasize}")
    base.mkdir(parents=True, exist_ok=True)
    per_model_dir = base / "per_model"
    per_model_dir.mkdir(exist_ok=True)
    print(f"\n  Output dir : {base}\n")

    for name, path in [(f"{IDD_TAG} images", IDD_IMAGES), (f"{IDD_TAG} labels", IDD_LABELS),
                       (f"{ODD_TAG} images", ODD_IMAGES), (f"{ODD_TAG} labels", ODD_LABELS)]:
        if not path.exists():
            print(f"[ERROR] {name} not found: {path}"); return

    idd_images = get_image_files(IDD_IMAGES)
    odd_images = get_image_files(ODD_IMAGES)
    print(f"  {IDD_TAG} : {len(idd_images)} images")
    print(f"  {ODD_TAG} : {len(odd_images)} images")
    print(f"  IoU thr : {IOU_THRESHOLD}  |  Conf : {CONF_THRESHOLDS}\n")

    print("  Pre-loading GT caches...", flush=True)
    idd_gt = preload_gt_cache(idd_images, IDD_IMAGES, IDD_LABELS)
    odd_gt = preload_gt_cache(odd_images, ODD_IMAGES, ODD_LABELS, ODD_CLASS_REMAP)
    print(f"  IDD GT: {len(idd_gt)}  ODD GT: {len(odd_gt)}\n")

    all_results = {}
    t_total     = time.time()

    for config in configs:
        wpath     = RUNS_DIR / f"{config}_{args.datasize}_yolo11{args.size}_adamw_100ep{_SEED_SUFFIX}" / "weights" / "best.pt"
        json_path = per_model_dir / f"{config}.json"

        if args.resume and json_path.exists():
            print(f"  [RESUME] {config}")
            all_results[config] = json.loads(json_path.read_text())
            continue

        if not wpath.exists():
            print(f"  [SKIP] {config}: {wpath}"); continue

        print(f"\n{'='*60}\n  {configs.index(config)+1}/{len(configs)}  {config}  (yolo11{args.size}, {args.datasize})\n{'='*60}")
        t0      = time.time()
        result  = evaluate_model(config, wpath, idd_images, idd_gt, odd_images, odd_gt)
        elapsed = (time.time() - t0) / 60
        all_results[config] = result
        json_path.parent.mkdir(parents=True, exist_ok=True)   # guard against missing dir
        json_path.write_text(json.dumps(result, indent=2))
        print(f"\n  Saved -> {json_path}  ({elapsed:.1f} min)")

    print(f"\n{'='*60}")
    print(f"  Done in {(time.time()-t_total)/60:.1f} min  |  Writing outputs...")
    print(f"{'='*60}\n")

    grid_str = make_macroF1_grid(all_results)
    (base / "table_macroF1.txt").write_text(grid_str)
    print(grid_str)

    (base / "report.txt").write_text(make_report(all_results))
    print(f"  Report  -> {base / 'report.txt'}")

    export_csv(all_results, base / "summary.csv")

    manifest = {
        "run_timestamp": ts, "iou_threshold": IOU_THRESHOLD,
        "model_size": f"yolo11{args.size}", "data_size": args.datasize,
        "conf_thresholds": CONF_THRESHOLDS,
        "idd_split": f"ITA.yolo/{IDD_SPLIT}", "odd_split": f"USA.yolo/{ODD_SPLIT}",
        "idd_images": len(idd_images), "odd_images": len(odd_images),
        "configs_evaluated": list(all_results.keys()),
        "configs_missing": [c for c in configs if c not in all_results],
    }
    (base / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  DONE -> {base}\n")


if __name__ == "__main__":
    main()
