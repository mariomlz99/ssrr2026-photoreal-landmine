#!/usr/bin/env python3
"""
SSRR 2026 - IoU threshold sensitivity analysis.

Purpose: demonstrate that the synthetic-over-real advantage is NOT an artifact
of the loose IoU=0.001 matching threshold used throughout the ablation.

Strategy: run GPU inference ONCE per model (conf=0.25), cache raw predictions,
then sweep IoU thresholds in pure numpy. This avoids repeating expensive GPU
work for each IoU value.

Splits (must match the two eval scripts):
  IDD : ITA.yolo / test   (held out; real model tuned best.pt on val)
  ODD : USA.yolo / val    (only USA split = the OOD test set)

Models: ALL 14 synthetic configs + real_vv by default (no selection bias).
The AUC table and rank_check cover every model; the paper FIGURE plots only a
readable subset (top-N synthetic by ODD MF1@0.001 + baseline + real), and
figure_subset.txt documents exactly which lines were plotted and why.

IoU sweep: 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50

Outputs (eval_results/iou_sensitivity/<run-timestamp>/):
  results.csv         macro_f1, pfm1_{f1,recall}, starfish_{f1,recall}
                      per (model, domain, iou_threshold)  -- ALL models
  auc.csv / auc.txt   AUC under MF1-vs-IoU curve [0.05, 0.50] per (model, domain)
  rank_check.txt      synth-vs-real ordering across the sweep, BOTH domains,
                      with crossing detection (ODD = headline; IDD = competitive)
  figure_subset.txt   which models were plotted and the objective selection rule
  fig_odd.pdf         Line plot: ODD macro-F1 vs IoU (paper figure)
  fig_idd.pdf         Line plot: IDD macro-F1 vs IoU
  fig_combined.pdf    2-panel figure for paper (ODD left, IDD right)

Usage:
    python eval_iou_sensitivity.py                       # all 14 synth + real
    python eval_iou_sensitivity.py --models t_10 n_0 baseline real_vv
    python eval_iou_sensitivity.py --resume
    python eval_iou_sensitivity.py --no-figs   # skip matplotlib if not installed
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
SULAND_ROOT  = Path(os.environ.get("SSRR_DATA",str(Path(__file__).resolve().parents[1] / "SULAND")))
SSRR_ROOT    = Path(os.environ.get("SSRR_RUNS_BASE",str(Path(__file__).resolve().parents[1])))
SYNTH_RUNS   = SSRR_ROOT / "runs" / "detect" / "synthetic_ablation"
REAL_RUNS    = SSRR_ROOT / "runs" / "detect" / "real_ablation"
RESULTS_BASE = SSRR_ROOT / "eval_results" / "iou_sensitivity"

# IDD uses the held-out TEST split: the real model selected best.pt on ITA-val,
# so scoring it on val would inflate its IDD number. test is unseen by every model.
# Must match eval_ablation_suland.py / eval_real_ablation_suland.py.
IDD_SPLIT  = "test"
ODD_SPLIT  = "val"
IDD_IMAGES = SULAND_ROOT / "iid" / "ITA.yolo" / IDD_SPLIT / "images"
IDD_LABELS = SULAND_ROOT / "iid" / "ITA.yolo" / IDD_SPLIT / "labels"
ODD_IMAGES = SULAND_ROOT / "ood" / "USA.yolo" / ODD_SPLIT / "images"
ODD_LABELS = SULAND_ROOT / "ood" / "USA.yolo" / ODD_SPLIT / "labels"
IDD_TAG    = f"IDD-{IDD_SPLIT}"
ODD_TAG    = f"ODD-{ODD_SPLIT}"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["pfm1", "starfish"]
NC          = len(CLASS_NAMES)
CONF        = 0.25       # fixed confidence for IoU sweep
DEVICE      = "0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 256
HALF        = torch.cuda.is_available()
IMGSZ       = 640
VALID_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ODD_CLASS_REMAP = {0: 1, 1: 0}

IOU_SWEEP = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# AUC integration range (standard-ish IoU range, excludes near-zero pathology)
AUC_IOU_MIN = 0.05
AUC_IOU_MAX = 0.50

# Sweep ALL synthetic models by default (no selection bias - most rigorous).
# AUC + rank_check cover every model; the FIGURE plots only a readable subset.
ALL_SYNTH_MODELS = [
    "baseline", "v_low", "v_mid", "h_off", "s_off", "i_1",
    "t_10", "t_5", "n_0", "n_30", "inv_off", "sun_off", "yaw_off", "pen_off",
]
DEFAULT_SYNTH_MODELS = ALL_SYNTH_MODELS
DEFAULT_REAL_MODELS  = ["real_vv"]

# Figure shows top-N synthetic (ranked by ODD macro-F1 at IoU=0.001, the
# canonical operating point) + baseline + real. Everything else stays in the
# CSV/AUC/rank_check tables.
N_FIG_SYNTH = 3

CONFIG_LABELS = {
    "baseline": "Baseline",
    "v_low":    "V-Low",   "v_mid":  "V-Mid",
    "h_off":    "H-Off",   "s_off":  "S-Off",
    "i_1":      "I-1",
    "t_10":     "T-10",    "t_5":    "T-5",
    "n_0":      "N-0",     "n_30":   "N-30",
    "inv_off":  "Inv-Off", "sun_off":"Sun-Off",
    "yaw_off":  "Yaw-Off", "pen_off":"Pen-Off",
    "real_vv":  "Real IDD",
}

# Colours for figures (consistent across scripts)
MODEL_COLORS = {
    "t_10":     "#1a7d3b",   # deep green
    "n_0":      "#5ab56e",   # mid green
    "baseline": "#2c5f8e",   # blue
    "real_vv":  "#c0392b",   # red
}
MODEL_LINESTYLES = {
    "t_10":     "-",
    "n_0":      "--",
    "baseline": "-.",
    "real_vv":  ":",
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
        rel  = Path(img_path).relative_to(images_root)
        lbl  = Path(labels_root) / rel.with_suffix(".txt")
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


def gt_to_pixels(img_path: str, cache: dict, shape: tuple) -> list:
    h, w = shape
    return [(cls, (xc-nw/2)*w, (yc-nh/2)*h, (xc+nw/2)*w, (yc+nh/2)*h)
            for cls, xc, yc, nw, nh in cache.get(img_path, [])]


# ---------------------------------------------------------------------------
# Raw prediction cache (run GPU inference once, reuse for all IoU thresholds)
# ---------------------------------------------------------------------------

def cache_raw_predictions(model, images: list, gt_cache: dict, tag: str) -> list:
    """
    Returns list of dicts, one per image:
      {
        "gt":   [(cls, x1,y1,x2,y2), ...],   # pixel coords
        "pred": [(cls, x1,y1,x2,y2), ...],   # pixel coords, conf already filtered
      }
    """
    records = []
    total   = len(images)
    print(f"\n  Caching predictions: {tag}  ({total} images, batch={BATCH_SIZE}, conf={CONF})")
    for start in range(0, total, BATCH_SIZE):
        batch   = images[start:start+BATCH_SIZE]
        results = model.predict(
            source=batch, conf=CONF, batch=len(batch),
            device=DEVICE, half=HALF, imgsz=IMGSZ, verbose=False,
        )
        for img_path, pred in zip(batch, results):
            gt_boxes = gt_to_pixels(img_path, gt_cache, pred.orig_shape)
            if pred.boxes is not None and len(pred.boxes) > 0:
                pred_boxes = list(zip(
                    pred.boxes.cls.cpu().numpy().astype(np.int32),
                    pred.boxes.xyxy.cpu().numpy().tolist(),
                ))
                pred_list = [(int(cls), *box) for cls, box in pred_boxes]
            else:
                pred_list = []
            records.append({"gt": gt_boxes, "pred": pred_list})
        done = min(start + len(batch), total)
        print(f"\r  cached {done}/{total}", end="", flush=True)
    print()
    return records


# ---------------------------------------------------------------------------
# Confusion matrix from cached records at a given IoU threshold
# ---------------------------------------------------------------------------

def build_matrix(records: list, iou_thr: float) -> np.ndarray:
    matrix = np.zeros((NC+1, NC+1), dtype=np.float64)
    for rec in records:
        gt_boxes   = rec["gt"]    # list of (cls, x1,y1,x2,y2)
        pred_list  = rec["pred"]  # list of (cls, x1,y1,x2,y2)

        if not pred_list:
            for cls, *_ in gt_boxes:
                matrix[cls, NC] += 1
            continue

        pboxes = np.array([[x1,y1,x2,y2] for _,x1,y1,x2,y2 in pred_list], dtype=np.float32)
        pcls   = np.array([c for c,*_ in pred_list], dtype=np.int32)
        P      = len(pcls)

        if not gt_boxes:
            for pi in range(P):
                matrix[NC, pcls[pi]] += 1
            continue

        G      = len(gt_boxes)
        gboxes = np.array([[x1,y1,x2,y2] for _,x1,y1,x2,y2 in gt_boxes], dtype=np.float32)
        gcls   = np.array([c for c,*_ in gt_boxes], dtype=np.int32)

        xl = np.maximum(gboxes[:,0:1], pboxes[:,0])
        yt = np.maximum(gboxes[:,1:2], pboxes[:,1])
        xr = np.minimum(gboxes[:,2:3], pboxes[:,2])
        yb = np.minimum(gboxes[:,3:4], pboxes[:,3])
        inter   = np.maximum(0.0, xr-xl) * np.maximum(0.0, yb-yt)
        ag      = (gboxes[:,2]-gboxes[:,0]) * (gboxes[:,3]-gboxes[:,1])
        ap      = (pboxes[:,2]-pboxes[:,0]) * (pboxes[:,3]-pboxes[:,1])
        union   = ag[:,None] + ap[None,:] - inter
        iou_mat = np.where(union > 0.0, inter/union, 0.0)

        used = np.zeros(P, dtype=bool)
        for g in range(G):
            row = iou_mat[g].copy(); row[used] = -1.0
            bi  = int(np.argmax(row))
            if row[bi] >= iou_thr:
                matrix[gcls[g], pcls[bi]] += 1
                used[bi] = True
            else:
                matrix[gcls[g], NC] += 1
        for pi in range(P):
            if not used[pi]:
                matrix[NC, pcls[pi]] += 1

    return matrix


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
            "precision": round(prec, 6), "recall": round(rec, 6), "f1": round(f1, 6),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "support": int(support),
        }
        f1s.append(f1); precs.append(prec); recs.append(rec)
    return {
        "classes":      classes,
        "macro_f1":     round(float(np.mean(f1s)),   6),
        "macro_recall": round(float(np.mean(recs)),  6),
        "macro_prec":   round(float(np.mean(precs)), 6),
    }


# ---------------------------------------------------------------------------
# AUC under MF1-vs-IoU curve (trapezoidal, over [AUC_IOU_MIN, AUC_IOU_MAX])
# ---------------------------------------------------------------------------

def compute_auc(iou_vals: list, mf1_vals: list) -> float:
    pairs = [(iou, mf1) for iou, mf1 in zip(iou_vals, mf1_vals)
             if AUC_IOU_MIN <= iou <= AUC_IOU_MAX]
    if len(pairs) < 2:
        return float("nan")
    pairs.sort(key=lambda x: x[0])
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    # manual trapezoid (np.trapz was removed in numpy 2.0)
    area = sum((xs[i+1]-xs[i]) * (ys[i+1]+ys[i]) / 2.0 for i in range(len(xs)-1))
    return round(area / (xs[-1] - xs[0]), 6)


# ---------------------------------------------------------------------------
# Resolve weight path for a given model name
# ---------------------------------------------------------------------------

def weight_path_for(model_name: str, size: str = "l", datasize: str = "30k") -> Path:
    if model_name.startswith("real"):
        # real has NO data-size axis (trained on the fixed real set); datasize ignored
        d = REAL_RUNS / f"real_idd_yolo11{size}_adamw_100ep"
        # L was copied as a flat best.pt; freshly-trained n/s live under weights/
        return d / "weights" / "best.pt" if (d / "weights" / "best.pt").exists() else d / "best.pt"
    return SYNTH_RUNS / f"{model_name}_{datasize}_yolo11{size}_adamw_100ep" / "weights" / "best.pt"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IoU threshold sensitivity analysis")
    parser.add_argument("--models", nargs="+",
                        default=DEFAULT_SYNTH_MODELS + DEFAULT_REAL_MODELS,
                        help="Model names to evaluate (e.g. t_10 n_0 baseline real_vv)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip models whose raw prediction cache already exists")
    parser.add_argument("--no-figs", action="store_true",
                        help="Skip figure generation (useful if matplotlib not available)")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--size", default="l", choices=["l", "s", "n"],
                        help="YOLO11 model size to evaluate (default: l)")
    parser.add_argument("--datasize", default="30k", choices=["10k", "20k", "30k"],
                        help="Synthetic dataset size (real ignores this) (default: 30k)")
    args = parser.parse_args()

    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(args.outdir) if args.outdir else RESULTS_BASE / f"run_{ts}_yolo11{args.size}_{args.datasize}"
    base.mkdir(parents=True, exist_ok=True)
    cache_dir = base / "pred_cache"
    cache_dir.mkdir(exist_ok=True)
    print(f"\n  Output dir : {base}")
    print(f"  Models     : {args.models}")
    print(f"  IoU sweep  : {IOU_SWEEP}")
    print(f"  Conf (fixed): {CONF}\n")

    # Image lists and GT caches
    idd_images = get_image_files(IDD_IMAGES)
    odd_images = get_image_files(ODD_IMAGES)

    print(f"  {IDD_TAG} : {len(idd_images)} images")
    print(f"  {ODD_TAG} : {len(odd_images)} images\n")

    print("  Pre-loading GT label caches...")
    idd_gt = preload_gt_cache(idd_images, IDD_IMAGES, IDD_LABELS)
    odd_gt = preload_gt_cache(odd_images, ODD_IMAGES, ODD_LABELS, ODD_CLASS_REMAP)
    print(f"  IDD GT: {len(idd_gt)} entries  |  ODD GT: {len(odd_gt)} entries\n")

    # GPU pass: cache raw predictions per model
    raw_cache = {}  # model_name -> {"idd": [...], "odd": [...]}

    for model_name in args.models:
        wpath = weight_path_for(model_name, args.size, args.datasize)
        if not wpath.exists():
            print(f"  [SKIP] {model_name}: no weights at {wpath}")
            continue

        cache_path = cache_dir / f"{model_name}.json"

        if args.resume and cache_path.exists():
            print(f"  [RESUME] Loading prediction cache for {model_name}")
            raw_cache[model_name] = json.loads(cache_path.read_text())
            continue

        print(f"\n  {'='*56}")
        print(f"  Loading model: {model_name}  ({CONFIG_LABELS.get(model_name, model_name)})")
        print(f"  Weights: {wpath}")
        print(f"  {'='*56}")

        model = YOLO(str(wpath))
        t0    = time.time()

        idd_records = cache_raw_predictions(model, idd_images, idd_gt, f"{model_name}/{IDD_TAG}")
        odd_records = cache_raw_predictions(model, odd_images, odd_gt, f"{model_name}/{ODD_TAG}")

        elapsed = (time.time() - t0) / 60
        print(f"  Inference done in {elapsed:.1f} min")

        raw_cache[model_name] = {"idd": idd_records, "odd": odd_records}
        cache_path.write_text(json.dumps(raw_cache[model_name]))
        print(f"  Cached -> {cache_path}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if not raw_cache:
        print("\n[ERROR] No model caches available. Check weights paths.")
        return

    # IoU sweep - pure numpy, no GPU
    print(f"\n  {'='*56}")
    print(f"  Sweeping {len(IOU_SWEEP)} IoU thresholds in numpy...")
    print(f"  {'='*56}\n")

    # results[model_name][domain][iou_thr] = metrics dict
    results = {m: {"idd": {}, "odd": {}} for m in raw_cache}

    for model_name, cache in raw_cache.items():
        for dom in ["idd", "odd"]:
            records = cache[dom]
            for iou_thr in IOU_SWEEP:
                mat     = build_matrix(records, iou_thr)
                metrics = matrix_metrics(mat)
                results[model_name][dom][iou_thr] = metrics
            mf1_at_001 = results[model_name][dom][0.001]["macro_f1"]
            mf1_at_050 = results[model_name][dom][0.50]["macro_f1"]
            print(f"  {model_name:<12} {dom.upper():<5}  "
                  f"MF1@0.001={mf1_at_001:.3f}  MF1@0.50={mf1_at_050:.3f}  "
                  f"drop={mf1_at_001-mf1_at_050:.3f}")

    # Export results CSV
    csv_rows = []
    for model_name in raw_cache:
        for dom in ["idd", "odd"]:
            for iou_thr in IOU_SWEEP:
                m  = results[model_name][dom][iou_thr]
                p1 = m["classes"]["pfm1"]
                sf = m["classes"]["starfish"]
                csv_rows.append({
                    "model":           model_name,
                    "model_label":     CONFIG_LABELS.get(model_name, model_name),
                    "domain":          dom,
                    "iou_threshold":   iou_thr,
                    "macro_f1":        m["macro_f1"],
                    "macro_recall":    m["macro_recall"],
                    "macro_prec":      m["macro_prec"],
                    "pfm1_f1":         p1["f1"],
                    "pfm1_recall":     p1["recall"],
                    "pfm1_precision":  p1["precision"],
                    "pfm1_tp":         p1["tp"],
                    "pfm1_fp":         p1["fp"],
                    "pfm1_fn":         p1["fn"],
                    "starfish_f1":        sf["f1"],
                    "starfish_recall":    sf["recall"],
                    "starfish_precision": sf["precision"],
                    "starfish_tp":        sf["tp"],
                    "starfish_fp":        sf["fp"],
                    "starfish_fn":        sf["fn"],
                })

    csv_path = base / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader()
        w.writerows(csv_rows)
    print(f"\n  Results CSV -> {csv_path}")

    # AUC table
    auc_rows  = []
    auc_lines = [
        f"\n  AUC under MF1-vs-IoU curve [{AUC_IOU_MIN}, {AUC_IOU_MAX}]",
        f"  (normalised trapezoid integral)",
        f"  {'Model':<14} {IDD_TAG:>10} {ODD_TAG:>10}",
        "  " + "-" * 38,
    ]
    for model_name in raw_cache:
        auc_idd = compute_auc(IOU_SWEEP,
                              [results[model_name]["idd"][t]["macro_f1"] for t in IOU_SWEEP])
        auc_odd = compute_auc(IOU_SWEEP,
                              [results[model_name]["odd"][t]["macro_f1"] for t in IOU_SWEEP])
        auc_lines.append(f"  {CONFIG_LABELS.get(model_name, model_name):<14} {auc_idd:>10.4f} {auc_odd:>10.4f}")
        auc_rows.append({"model": model_name, "model_label": CONFIG_LABELS.get(model_name, model_name),
                         "auc_idd": auc_idd, "auc_odd": auc_odd})
    auc_lines.append("")

    auc_txt = "\n".join(auc_lines)
    (base / "auc.txt").write_text(auc_txt)
    print(auc_txt)

    auc_csv_path = base / "auc.csv"
    with open(auc_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","model_label","auc_idd","auc_odd"])
        w.writeheader()
        w.writerows(auc_rows)
    print(f"  AUC CSV   -> {auc_csv_path}")

    # Rank check: how the synthetic-vs-real ordering behaves as a function of the
    # IoU sweep, for BOTH domains. ODD = headline claim (expect all-preserved).
    # IDD = in-distribution where the real model is competitive, so the ordering
    # can CROSS as IoU rises -- we detect and report exactly where.
    real_models  = [m for m in raw_cache if m.startswith("real")]
    synth_models = [m for m in raw_cache if not m.startswith("real")]
    N_SYNTH      = len(synth_models)

    def _domain_rank_block(dom: str, dom_tag: str, note: str) -> tuple:
        """Returns (lines, all_preserved, crossings)."""
        lines = [
            f"\n  === {dom_tag} ({'USA' if dom=='odd' else 'ITA'}) -- {note} ===",
            f"  {'IoU':>6}  {'Real':>8}  {'BestSynth':>9}  {'WorstSynth':>10}"
            f"  {'#synth>real':>12}  {'all>real?':>9}",
            "  " + "-" * 64,
        ]
        all_pres = True
        prev_best_gt = None         # was best-synth > real at previous IoU?
        prev_count   = None         # how many synth > real previously
        crossings    = []
        for iou_thr in IOU_SWEEP:
            real_mf1   = results[real_models[0]][dom][iou_thr]["macro_f1"] if real_models else float("nan")
            synth_mf1s = [results[m][dom][iou_thr]["macro_f1"] for m in synth_models]
            best, worst = max(synth_mf1s), min(synth_mf1s)
            count_gt   = sum(1 for s in synth_mf1s if s > real_mf1) if real_models else N_SYNTH
            all_gt     = (count_gt == N_SYNTH)
            if not all_gt:
                all_pres = False
            best_gt = best > real_mf1 if real_models else True
            # detect crossings vs previous threshold
            if prev_best_gt is not None and best_gt != prev_best_gt:
                crossings.append(f"best-synth {'overtakes' if best_gt else 'falls below'} real at IoU={iou_thr}")
            if prev_count is not None and count_gt != prev_count:
                crossings.append(f"#synth>real {prev_count}->{count_gt} at IoU={iou_thr}")
            prev_best_gt, prev_count = best_gt, count_gt
            lines.append(
                f"  {iou_thr:>6.3f}  {real_mf1:>8.4f}  {best:>9.4f}  {worst:>10.4f}"
                f"  {count_gt:>7}/{N_SYNTH:<4}  {('YES' if all_gt else '** NO **'):>9}"
            )
        lines.append("  " + "-" * 64)
        if crossings:
            lines.append(f"  Crossings: " + " | ".join(crossings))
        else:
            lines.append(f"  Crossings: none (ordering stable across the whole sweep)")
        lines.append(f"  Overall {dom_tag}: "
                     f"{'ALL '+str(N_SYNTH)+' synth > real at EVERY IoU' if all_pres else 'ranking breaks at some IoU'}")
        return lines, all_pres, crossings

    rank_lines = [
        "\n  RANKING vs. IoU SWEEP  (synthetic vs. real)",
        f"  Real model   : {real_models}",
        f"  Synth models : {N_SYNTH} configs",
    ]
    odd_block, odd_pres, _ = _domain_rank_block("odd", ODD_TAG, "domain shift; HEADLINE claim")
    idd_block, idd_pres, idd_cross = _domain_rank_block("idd", IDD_TAG, "in-distribution; real model competitive")
    rank_lines += odd_block + idd_block + ["", ""]
    rank_txt = "\n".join(rank_lines)
    (base / "rank_check.txt").write_text(rank_txt)
    print(rank_txt)

    # Figure subset: top-N synthetic by ODD MF1@0.001 + baseline + real.
    # Selection is objective and derived from THIS run's results (not hardcoded).
    synth_models = [m for m in raw_cache if not m.startswith("real")]
    real_models  = [m for m in raw_cache if m.startswith("real")]
    ranked_odd   = sorted(
        synth_models,
        key=lambda m: results[m]["odd"][0.001]["macro_f1"],
        reverse=True,
    )
    top_synth = [m for m in ranked_odd if m != "baseline"][:N_FIG_SYNTH]
    fig_subset = top_synth \
        + (["baseline"] if "baseline" in raw_cache else []) \
        + real_models

    subset_lines = [
        "\n  FIGURE SUBSET (plotted lines)",
        f"  Rule: top-{N_FIG_SYNTH} synthetic by ODD macro-F1 @ IoU=0.001  +  baseline  +  real",
        "",
        f"  {'rank':>4}  {'model':<12} {'ODD MF1@.001':>13} {'IDD MF1@.001':>13}  plotted",
        "  " + "-" * 56,
    ]
    for i, m in enumerate(ranked_odd):
        odd = results[m]["odd"][0.001]["macro_f1"]
        idd = results[m]["idd"][0.001]["macro_f1"]
        mark = "YES" if m in fig_subset else ""
        subset_lines.append(f"  {i+1:>4}  {CONFIG_LABELS.get(m,m):<12} {odd:>13.4f} {idd:>13.4f}  {mark}")
    for m in real_models:
        odd = results[m]["odd"][0.001]["macro_f1"]
        idd = results[m]["idd"][0.001]["macro_f1"]
        subset_lines.append(f"  {'real':>4}  {CONFIG_LABELS.get(m,m):<12} {odd:>13.4f} {idd:>13.4f}  YES")
    subset_lines.append("")
    subset_txt = "\n".join(subset_lines)
    (base / "figure_subset.txt").write_text(subset_txt)
    print(subset_txt)

    # Figures
    if not args.no_figs:
        try:
            _make_figures(results, raw_cache, base, fig_subset)
        except ImportError as e:
            print(f"  [WARN] Could not generate figures: {e}")
            print(f"         Re-run without --no-figs once matplotlib is available.")

    print(f"\n  DONE -- outputs in {base}\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _make_figures(results: dict, raw_cache: dict, base: Path, fig_subset: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    FONT_SIZE   = 11
    LEGEND_SIZE = 9.5
    LW          = 2.0
    MARKER_SIZE = 5

    plt.rcParams.update({
        "font.family":    "DejaVu Sans",
        "font.size":      FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize":FONT_SIZE - 1,
        "ytick.labelsize":FONT_SIZE - 1,
        "legend.fontsize":LEGEND_SIZE,
        "figure.dpi":     150,
    })

    # Fixed styles for known anchors; dynamic palette for the rest of the subset.
    PALETTE = ["#1a7d3b", "#5ab56e", "#8e44ad", "#16a085", "#d35400", "#2980b9"]
    MARK_CYCLE = ["o", "s", "^", "v", "P", "X", "*"]
    colors, markers, lss = {}, {}, {}
    pi = 0
    for m in fig_subset:
        if m in MODEL_COLORS:
            colors[m] = MODEL_COLORS[m]
        elif m == "baseline":
            colors[m] = "#2c5f8e"
        elif m.startswith("real"):
            colors[m] = "#c0392b"
        else:
            colors[m] = PALETTE[pi % len(PALETTE)]; pi += 1
        lss[m]     = MODEL_LINESTYLES.get(m, ":" if m.startswith("real") else "-")
        markers[m] = {"baseline": "^", "real_vv": "D"}.get(m, MARK_CYCLE[fig_subset.index(m) % len(MARK_CYCLE)])

    def _plot_domain(ax, dom: str, title: str) -> None:
        ax.set_title(title, fontweight="bold")
        for model_name in fig_subset:
            mf1s  = [results[model_name][dom][t]["macro_f1"] for t in IOU_SWEEP]
            label = CONFIG_LABELS.get(model_name, model_name)
            c     = colors.get(model_name, "#888888")
            ls    = lss.get(model_name, "-")
            mk    = markers.get(model_name, "x")
            zord  = 3 if model_name.startswith("real") else 4
            ax.plot(IOU_SWEEP, mf1s, color=c, linestyle=ls, linewidth=LW,
                    marker=mk, markersize=MARKER_SIZE, label=label, zorder=zord)

        # Shade gap between best synth and real at each threshold
        real_models  = [m for m in fig_subset if m.startswith("real")]
        synth_models = [m for m in fig_subset if not m.startswith("real")]
        if real_models and synth_models:
            real_mf1s   = [results[real_models[0]][dom][t]["macro_f1"] for t in IOU_SWEEP]
            best_synth  = [max(results[m][dom][t]["macro_f1"] for m in synth_models)
                           for t in IOU_SWEEP]
            ax.fill_between(IOU_SWEEP, real_mf1s, best_synth,
                            alpha=0.08, color="#1a7d3b", label="_synth_advantage")

        ax.set_xlabel("IoU threshold")
        ax.set_ylabel("Macro-F1")
        ax.set_xlim(-0.01, 0.52)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.set_xticks([0.001, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
        ax.set_xticklabels(["0.001", "0.05", "0.10", "0.20", "0.30", "0.40", "0.50"], rotation=30)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        ax.legend(loc="upper right")

    # Individual domain figures
    for dom, fname, title in [
        ("odd", "fig_odd.pdf",  f"ODD (USA, {ODD_SPLIT}) -- Macro-F1 vs. IoU threshold"),
        ("idd", "fig_idd.pdf",  f"IDD (ITA, {IDD_SPLIT}) -- Macro-F1 vs. IoU threshold"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _plot_domain(ax, dom, title)
        fig.tight_layout()
        fig.savefig(base / fname, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure -> {base / fname}")

    # Combined 2-panel figure for paper
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    _plot_domain(axes[0], "odd", f"(a) ODD (USA, {ODD_SPLIT}) -- domain shift")
    _plot_domain(axes[1], "idd", f"(b) IDD (ITA, {IDD_SPLIT}) -- in-distribution")
    # Remove duplicate legends
    axes[1].get_legend().remove()
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.suptitle("Macro-F1 vs. IoU threshold (conf=0.25)", fontsize=12, fontweight="bold")
    fig.savefig(base / "fig_combined.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure -> {base / 'fig_combined.pdf'}")

    # Per-class recall breakdown at conf=0.25, IoU=0.001 (summary bar)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, dom, dom_title in [
        (axes[0], "odd", "ODD -- Recall by class at IoU=0.001"),
        (axes[1], "idd", "IDD -- Recall by class at IoU=0.001"),
    ]:
        model_names = list(fig_subset)
        labels      = [CONFIG_LABELS.get(m, m) for m in model_names]
        pfm1_recs   = [results[m][dom][0.001]["classes"]["pfm1"]["recall"] for m in model_names]
        sfsh_recs   = [results[m][dom][0.001]["classes"]["starfish"]["recall"] for m in model_names]
        x  = np.arange(len(model_names))
        bw = 0.35
        bars1 = ax.bar(x - bw/2, pfm1_recs, bw, label="pfm1",    color="#2c5f8e", alpha=0.85)
        bars2 = ax.bar(x + bw/2, sfsh_recs, bw, label="starfish", color="#e67e22", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1.05)
        ax.set_title(dom_title, fontweight="bold")
        ax.legend()
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(base / "fig_perclass_recall.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure -> {base / 'fig_perclass_recall.pdf'}")


if __name__ == "__main__":
    main()
