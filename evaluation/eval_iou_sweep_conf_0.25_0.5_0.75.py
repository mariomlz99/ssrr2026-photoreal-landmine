#!/usr/bin/env python3
"""
SSRR 2026 -- IoU x Confidence sensitivity analysis (conf = 0.25 / 0.50 / 0.75).

Companion to eval_iou_sensitivity.py (which sweeps IoU at the single operating
point conf=0.25). This version additionally sweeps the CONFIDENCE threshold, so
you can see whether the synthetic-over-real conclusion -- and absolute
performance -- degrade along BOTH axes:
  * tighter IoU  (localization strictness)
  * higher conf  (precision/recall operating point)

FROM SCRATCH: inference is run SEPARATELY at each confidence level (one
model.predict per conf), not derived by filtering a single low-conf pass. This
costs 3x the GPU time but needs zero assumptions about NMS / max_det ordering,
so it is fully defensible in the paper. (For the record, filtering a conf=0.25
superset is provably identical because YOLO applies the conf threshold BEFORE
NMS and NMS only suppresses in favour of higher-scoring boxes -- but we run it
the explicit way here.)

Splits (identical to the two eval scripts):
  IDD : ITA.yolo / test   (held out; real model tuned best.pt on val)
  ODD : USA.yolo / val    (only USA split = the OOD test set)

Models: ALL 14 synthetic configs + real_vv by default. AUC + rank_check cover
every model; the figures plot a readable subset (top-N synth by ODD MF1@0.001 +
baseline + real), documented in figure_subset.txt.

Outputs (eval_results/iou_conf_sweep/<run-timestamp>/):
  results.csv            metrics per (model, domain, conf, iou_threshold)
  auc.csv / auc.txt      AUC under MF1-vs-IoU curve [0.05,0.50] per (model,domain,conf)
  rank_check.txt         synth-vs-real ordering across the IoU sweep, per conf,
                         BOTH domains, with crossing detection
  conf_degradation.csv   MF1 at the reference IoU vs conf (the degradation table)
  figure_subset.txt      which models are plotted + the objective selection rule
  fig_grid.pdf           3 (conf) x 2 (domain) grid of MF1-vs-IoU
  fig_combined_c025.pdf  per-conf 2-panel (ODD|IDD) MF1-vs-IoU
  fig_combined_c050.pdf
  fig_combined_c075.pdf
  fig_conf_degradation.pdf   MF1 vs conf at the reference IoU (both domains)

Usage:
    python eval_iou_sweep_conf_0.25_0.5_0.75.py
    python eval_iou_sweep_conf_0.25_0.5_0.75.py --models t_10 n_0 baseline real_vv
    python eval_iou_sweep_conf_0.25_0.5_0.75.py --resume
    python eval_iou_sweep_conf_0.25_0.5_0.75.py --no-figs
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
RESULTS_BASE = SSRR_ROOT / "eval_results" / "iou_conf_sweep"
_RUN_SEED    = int(os.environ.get("SSRR_RUN_SEED", 42))
_SEED_SUFFIX = "" if _RUN_SEED == 42 else f"_seed{_RUN_SEED}"   # resolve _seed{N} weights for alt seeds

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

CONF_LEVELS = [0.25, 0.50, 0.75]   # inference run separately at each

DEVICE      = "0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 256
HALF        = torch.cuda.is_available()
IMGSZ       = 640
VALID_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ODD_CLASS_REMAP = {0: 1, 1: 0}

IOU_SWEEP = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

# AUC integration range and the reference IoU used for ranking / degradation table
AUC_IOU_MIN = 0.05
AUC_IOU_MAX = 0.50
REF_IOU     = 0.001   # operating point: loosest matching (most adversarial)

ALL_SYNTH_MODELS = [
    "baseline", "v_low", "v_mid", "h_off", "s_off", "i_1",
    "t_10", "t_5", "n_0", "n_30", "inv_off", "sun_off", "yaw_off", "pen_off",
]
DEFAULT_SYNTH_MODELS = ALL_SYNTH_MODELS
DEFAULT_REAL_MODELS  = ["real_vv"]
N_FIG_SYNTH = 3   # top-N synthetic (by ODD MF1@REF_IOU, conf=0.25) shown in figures

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

MODEL_COLORS = {
    "t_10": "#1a7d3b", "n_0": "#5ab56e", "baseline": "#2c5f8e", "real_vv": "#c0392b",
}
MODEL_LINESTYLES = {"t_10": "-", "n_0": "--", "baseline": "-.", "real_vv": ":"}

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_image_files(root: Path):
    return sorted(str(p) for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in VALID_EXTS)


def preload_gt_cache(images, images_root, labels_root, remap=None):
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


def gt_to_pixels(img_path, cache, shape):
    h, w = shape
    return [(cls, (xc-nw/2)*w, (yc-nh/2)*h, (xc+nw/2)*w, (yc+nh/2)*h)
            for cls, xc, yc, nw, nh in cache.get(img_path, [])]


# ---------------------------------------------------------------------------
# Raw prediction cache -- inference at a SPECIFIC confidence level
# ---------------------------------------------------------------------------

def cache_raw_predictions(model, images, gt_cache, conf, tag):
    """
    One record per image, with predictions filtered by the network at `conf`:
      {"gt":   [(cls, x1,y1,x2,y2), ...],
       "pred": [(cls, x1,y1,x2,y2), ...]}
    """
    records = []
    total   = len(images)
    print(f"\n  Inference: {tag}  ({total} images, batch={BATCH_SIZE}, conf={conf})")
    for start in range(0, total, BATCH_SIZE):
        batch   = images[start:start+BATCH_SIZE]
        results = model.predict(source=batch, conf=conf, batch=len(batch),
                                device=DEVICE, half=HALF, imgsz=IMGSZ, verbose=False)
        for img_path, pred in zip(batch, results):
            gt_boxes = gt_to_pixels(img_path, gt_cache, pred.orig_shape)
            if pred.boxes is not None and len(pred.boxes) > 0:
                cls_a  = pred.boxes.cls.cpu().numpy().astype(np.int32)
                xyxy_a = pred.boxes.xyxy.cpu().numpy().tolist()
                pred_list = [(int(cls_a[i]), *xyxy_a[i]) for i in range(len(cls_a))]
            else:
                pred_list = []
            records.append({"gt": gt_boxes, "pred": pred_list})
        done = min(start + len(batch), total)
        print(f"\r  inferred {done}/{total}", end="", flush=True)
    print()
    return records


# ---------------------------------------------------------------------------
# Confusion matrix from cached records at a given IoU threshold
# ---------------------------------------------------------------------------

def build_matrix(records, iou_thr):
    matrix = np.zeros((NC+1, NC+1), dtype=np.float64)
    for rec in records:
        gt_boxes  = rec["gt"]
        pred_list = rec["pred"]

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


def matrix_metrics(matrix):
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


def compute_auc(iou_vals, mf1_vals):
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


def weight_path_for(model_name, size="l", datasize="30k"):
    if model_name.startswith("real"):
        # real has NO data-size axis (trained on the fixed real set); datasize ignored
        d = REAL_RUNS / f"real_idd_yolo11{size}_adamw_100ep{_SEED_SUFFIX}"
        # L was copied as a flat best.pt; freshly-trained n/s live under weights/
        return d / "weights" / "best.pt" if (d / "weights" / "best.pt").exists() else d / "best.pt"
    return SYNTH_RUNS / f"{model_name}_{datasize}_yolo11{size}_adamw_100ep{_SEED_SUFFIX}" / "weights" / "best.pt"


def cs(conf):
    return f"{conf:.2f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="IoU x Confidence sensitivity analysis")
    parser.add_argument("--models", nargs="+",
                        default=DEFAULT_SYNTH_MODELS + DEFAULT_REAL_MODELS)
    parser.add_argument("--resume", action="store_true",
                        help="Skip models whose raw prediction cache already exists")
    parser.add_argument("--no-figs", action="store_true")
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
    print(f"\n  Output dir  : {base}")
    print(f"  Models      : {args.models}")
    print(f"  Conf levels : {CONF_LEVELS}  (inference run separately at each)")
    print(f"  IoU sweep   : {IOU_SWEEP}\n")

    idd_images = get_image_files(IDD_IMAGES)
    odd_images = get_image_files(ODD_IMAGES)
    print(f"  {IDD_TAG} : {len(idd_images)} images")
    print(f"  {ODD_TAG} : {len(odd_images)} images\n")

    print("  Pre-loading GT label caches...")
    idd_gt = preload_gt_cache(idd_images, IDD_IMAGES, IDD_LABELS)
    odd_gt = preload_gt_cache(odd_images, ODD_IMAGES, ODD_LABELS, ODD_CLASS_REMAP)
    print(f"  {IDD_TAG} GT: {len(idd_gt)}  |  {ODD_TAG} GT: {len(odd_gt)}\n")

    # ----- GPU pass: inference at EACH conf level, cached per model -----
    # raw_cache[model][conf_str] = {"idd": [records], "odd": [records]}
    raw_cache = {}
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
        print(f"  {'='*56}")
        model = YOLO(str(wpath))
        t0    = time.time()
        per_conf = {}
        for conf in CONF_LEVELS:
            c = cs(conf)
            idd_records = cache_raw_predictions(model, idd_images, idd_gt, conf, f"{model_name}/{IDD_TAG}/c{c}")
            odd_records = cache_raw_predictions(model, odd_images, odd_gt, conf, f"{model_name}/{ODD_TAG}/c{c}")
            per_conf[c] = {"idd": idd_records, "odd": odd_records}
        print(f"  Inference done in {(time.time()-t0)/60:.1f} min  ({len(CONF_LEVELS)} conf levels)")
        raw_cache[model_name] = per_conf
        cache_path.write_text(json.dumps(per_conf))
        print(f"  Cached -> {cache_path}")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()

    if not raw_cache:
        print("\n[ERROR] No model caches available. Check weights paths.")
        return

    # ----- IoU sweep (pure numpy) over each (model, domain, conf) -----
    print(f"\n  {'='*56}")
    print(f"  Sweeping {len(IOU_SWEEP)} IoU thresholds x {len(CONF_LEVELS)} conf in numpy...")
    print(f"  {'='*56}\n")

    # results[model][dom][conf_str][iou] = metrics
    results = {m: {"idd": {}, "odd": {}} for m in raw_cache}
    for model_name, per_conf in raw_cache.items():
        for dom in ["idd", "odd"]:
            for conf in CONF_LEVELS:
                c = cs(conf)
                results[model_name][dom][c] = {}
                records = per_conf[c][dom]
                for iou_thr in IOU_SWEEP:
                    mat = build_matrix(records, iou_thr)
                    results[model_name][dom][c][iou_thr] = matrix_metrics(mat)
            line = "  ".join(
                f"c{cs(conf)}:MF1@{REF_IOU}={results[model_name][dom][cs(conf)][REF_IOU]['macro_f1']:.3f}"
                for conf in CONF_LEVELS)
            print(f"  {model_name:<12} {dom.upper():<5}  {line}")

    # ----- results.csv -----
    csv_rows = []
    for model_name in raw_cache:
        for dom in ["idd", "odd"]:
            for conf in CONF_LEVELS:
                c = cs(conf)
                for iou_thr in IOU_SWEEP:
                    m  = results[model_name][dom][c][iou_thr]
                    p1 = m["classes"]["pfm1"]; sf = m["classes"]["starfish"]
                    csv_rows.append({
                        "model": model_name, "model_label": CONFIG_LABELS.get(model_name, model_name),
                        "domain": dom, "conf": conf, "iou_threshold": iou_thr,
                        "macro_f1": m["macro_f1"], "macro_recall": m["macro_recall"],
                        "macro_prec": m["macro_prec"],
                        "pfm1_f1": p1["f1"], "pfm1_recall": p1["recall"], "pfm1_precision": p1["precision"],
                        "pfm1_tp": p1["tp"], "pfm1_fp": p1["fp"], "pfm1_fn": p1["fn"],
                        "starfish_f1": sf["f1"], "starfish_recall": sf["recall"], "starfish_precision": sf["precision"],
                        "starfish_tp": sf["tp"], "starfish_fp": sf["fp"], "starfish_fn": sf["fn"],
                    })
    csv_path = base / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        w.writeheader(); w.writerows(csv_rows)
    print(f"\n  Results CSV -> {csv_path}")

    # ----- AUC table (per model, domain, conf) -----
    auc_rows  = []
    auc_lines = [f"\n  AUC under MF1-vs-IoU curve [{AUC_IOU_MIN}, {AUC_IOU_MAX}] (normalised)"]
    for conf in CONF_LEVELS:
        c = cs(conf)
        auc_lines += [
            f"\n  --- conf = {c} ---",
            f"  {'Model':<14} {IDD_TAG:>10} {ODD_TAG:>10}",
            "  " + "-" * 38,
        ]
        for model_name in raw_cache:
            auc_idd = compute_auc(IOU_SWEEP, [results[model_name]["idd"][c][t]["macro_f1"] for t in IOU_SWEEP])
            auc_odd = compute_auc(IOU_SWEEP, [results[model_name]["odd"][c][t]["macro_f1"] for t in IOU_SWEEP])
            auc_lines.append(f"  {CONFIG_LABELS.get(model_name, model_name):<14} {auc_idd:>10.4f} {auc_odd:>10.4f}")
            auc_rows.append({"model": model_name, "model_label": CONFIG_LABELS.get(model_name, model_name),
                             "conf": conf, "auc_idd": auc_idd, "auc_odd": auc_odd})
    auc_txt = "\n".join(auc_lines) + "\n"
    (base / "auc.txt").write_text(auc_txt)
    print(auc_txt)
    with open(base / "auc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","model_label","conf","auc_idd","auc_odd"])
        w.writeheader(); w.writerows(auc_rows)
    print(f"  AUC CSV   -> {base / 'auc.csv'}")

    # ----- conf degradation table: MF1 at REF_IOU vs conf -----
    deg_rows  = []
    deg_lines = [f"\n  CONFIDENCE DEGRADATION  (Macro-F1 at IoU={REF_IOU})",
                 f"  {'Model':<14} {'Dom':<4}" + "".join(f"  {'c'+cs(c):>8}" for c in CONF_LEVELS)
                 + f"  {'drop .25->.75':>14}",
                 "  " + "-" * 60]
    for model_name in raw_cache:
        for dom in ["odd", "idd"]:
            vals = [results[model_name][dom][cs(c)][REF_IOU]["macro_f1"] for c in CONF_LEVELS]
            drop = vals[0] - vals[-1]
            deg_lines.append(f"  {CONFIG_LABELS.get(model_name, model_name):<14} {dom.upper():<4}"
                             + "".join(f"  {v:>8.4f}" for v in vals) + f"  {drop:>14.4f}")
            row = {"model": model_name, "domain": dom, "ref_iou": REF_IOU,
                   "drop_025_075": round(drop, 6)}
            for c, v in zip(CONF_LEVELS, vals):
                row[f"mf1_c{cs(c)}"] = v
            deg_rows.append(row)
    deg_txt = "\n".join(deg_lines) + "\n"
    (base / "conf_degradation_table.txt").write_text(deg_txt)
    print(deg_txt)
    with open(base / "conf_degradation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(deg_rows[0].keys()))
        w.writeheader(); w.writerows(deg_rows)
    print(f"  Degradation CSV -> {base / 'conf_degradation.csv'}")

    # ----- rank check: synth vs real across IoU, for EACH conf, BOTH domains -----
    real_models  = [m for m in raw_cache if m.startswith("real")]
    synth_models = [m for m in raw_cache if not m.startswith("real")]
    N_SYNTH      = len(synth_models)

    def _domain_rank_block(dom, dom_tag, conf, note):
        c = cs(conf)
        lines = [
            f"\n  --- {dom_tag} ({'USA' if dom=='odd' else 'ITA'}) @ conf={c}  [{note}] ---",
            f"  {'IoU':>6}  {'Real':>8}  {'BestSynth':>9}  {'WorstSynth':>10}"
            f"  {'#synth>real':>12}  {'all>real?':>9}",
            "  " + "-" * 64,
        ]
        all_pres = True
        prev_best_gt = prev_count = None
        crossings = []
        for iou_thr in IOU_SWEEP:
            real_mf1   = results[real_models[0]][dom][c][iou_thr]["macro_f1"] if real_models else float("nan")
            synth_mf1s = [results[m][dom][c][iou_thr]["macro_f1"] for m in synth_models]
            best, worst = max(synth_mf1s), min(synth_mf1s)
            count_gt   = sum(1 for s in synth_mf1s if s > real_mf1) if real_models else N_SYNTH
            all_gt     = (count_gt == N_SYNTH)
            if not all_gt:
                all_pres = False
            best_gt = best > real_mf1 if real_models else True
            if prev_best_gt is not None and best_gt != prev_best_gt:
                crossings.append(f"best-synth {'overtakes' if best_gt else 'falls below'} real at IoU={iou_thr}")
            if prev_count is not None and count_gt != prev_count:
                crossings.append(f"#synth>real {prev_count}->{count_gt} at IoU={iou_thr}")
            prev_best_gt, prev_count = best_gt, count_gt
            lines.append(f"  {iou_thr:>6.3f}  {real_mf1:>8.4f}  {best:>9.4f}  {worst:>10.4f}"
                         f"  {count_gt:>7}/{N_SYNTH:<4}  {('YES' if all_gt else '** NO **'):>9}")
        lines.append("  " + "-" * 64)
        lines.append("  Crossings: " + (" | ".join(crossings) if crossings else "none (stable across sweep)"))
        lines.append(f"  Overall: {'ALL '+str(N_SYNTH)+' synth > real at EVERY IoU' if all_pres else 'ranking breaks at some IoU'}")
        return lines

    rank_lines = ["\n  RANKING vs. IoU SWEEP, PER CONFIDENCE LEVEL  (synthetic vs. real)",
                  f"  Real model   : {real_models}",
                  f"  Synth models : {N_SYNTH} configs"]
    for conf in CONF_LEVELS:
        rank_lines += _domain_rank_block("odd", ODD_TAG, conf, "domain shift; HEADLINE")
        rank_lines += _domain_rank_block("idd", IDD_TAG, conf, "in-distribution; real competitive")
    rank_lines += ["", ""]
    (base / "rank_check.txt").write_text("\n".join(rank_lines))
    print("\n".join(rank_lines))

    # ----- figure subset (objective, from this run @ conf=0.25, REF_IOU) -----
    ranked_odd = sorted(synth_models,
                        key=lambda m: results[m]["odd"][cs(0.25)][REF_IOU]["macro_f1"],
                        reverse=True)
    top_synth  = [m for m in ranked_odd if m != "baseline"][:N_FIG_SYNTH]
    fig_subset = top_synth + (["baseline"] if "baseline" in raw_cache else []) + real_models

    subset_lines = [
        "\n  FIGURE SUBSET (plotted lines)",
        f"  Rule: top-{N_FIG_SYNTH} synthetic by ODD macro-F1 @ conf=0.25, IoU={REF_IOU}  +  baseline  +  real",
        "",
        f"  {'rank':>4}  {'model':<12} {'ODD MF1':>9} {'IDD MF1':>9}  plotted",
        "  " + "-" * 50,
    ]
    for i, m in enumerate(ranked_odd):
        odd = results[m]["odd"][cs(0.25)][REF_IOU]["macro_f1"]
        idd = results[m]["idd"][cs(0.25)][REF_IOU]["macro_f1"]
        subset_lines.append(f"  {i+1:>4}  {CONFIG_LABELS.get(m,m):<12} {odd:>9.4f} {idd:>9.4f}  {'YES' if m in fig_subset else ''}")
    for m in real_models:
        odd = results[m]["odd"][cs(0.25)][REF_IOU]["macro_f1"]
        idd = results[m]["idd"][cs(0.25)][REF_IOU]["macro_f1"]
        subset_lines.append(f"  {'real':>4}  {CONFIG_LABELS.get(m,m):<12} {odd:>9.4f} {idd:>9.4f}  YES")
    (base / "figure_subset.txt").write_text("\n".join(subset_lines) + "\n")
    print("\n".join(subset_lines))

    # ----- figures -----
    if not args.no_figs:
        try:
            _make_figures(results, raw_cache, base, fig_subset)
        except ImportError as e:
            print(f"  [WARN] Could not generate figures: {e}")

    print(f"\n  DONE -- outputs in {base}\n")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style_maps(fig_subset):
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
    return colors, markers, lss


def _make_figures(results, raw_cache, base, fig_subset):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 11,
        "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 9, "figure.dpi": 150,
    })
    colors, markers, lss = _style_maps(fig_subset)
    LW, MS = 2.0, 5

    def _plot(ax, dom, conf, title, legend=True):
        c = cs(conf)
        ax.set_title(title, fontweight="bold")
        for m in fig_subset:
            mf1s = [results[m][dom][c][t]["macro_f1"] for t in IOU_SWEEP]
            ax.plot(IOU_SWEEP, mf1s, color=colors[m], linestyle=lss[m], linewidth=LW,
                    marker=markers[m], markersize=MS, label=CONFIG_LABELS.get(m, m),
                    zorder=3 if m.startswith("real") else 4)
        real_m  = [m for m in fig_subset if m.startswith("real")]
        synth_m = [m for m in fig_subset if not m.startswith("real")]
        if real_m and synth_m:
            real_mf1   = [results[real_m[0]][dom][c][t]["macro_f1"] for t in IOU_SWEEP]
            best_synth = [max(results[m][dom][c][t]["macro_f1"] for m in synth_m) for t in IOU_SWEEP]
            ax.fill_between(IOU_SWEEP, real_mf1, best_synth, alpha=0.08, color="#1a7d3b")
        ax.set_xlabel("IoU threshold"); ax.set_ylabel("Macro-F1")
        ax.set_xlim(-0.01, 0.52); ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.001, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
        ax.set_xticklabels(["0.001","0.05","0.10","0.20","0.30","0.40","0.50"], rotation=30)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        if legend:
            ax.legend(loc="upper right")

    # 3 (conf) x 2 (domain) grid
    fig, axes = plt.subplots(len(CONF_LEVELS), 2, figsize=(13, 4.3*len(CONF_LEVELS)))
    for r, conf in enumerate(CONF_LEVELS):
        _plot(axes[r,0], "odd", conf, f"ODD (USA, {ODD_SPLIT}) -- conf={cs(conf)}", legend=(r==0))
        _plot(axes[r,1], "idd", conf, f"IDD (ITA, {IDD_SPLIT}) -- conf={cs(conf)}", legend=False)
    fig.suptitle("Macro-F1 vs. IoU threshold across confidence levels", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(base / "fig_grid.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"  Figure -> {base / 'fig_grid.pdf'}")

    # per-conf 2-panel
    for conf in CONF_LEVELS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        _plot(axes[0], "odd", conf, f"(a) ODD (USA, {ODD_SPLIT}) -- conf={cs(conf)}", legend=True)
        _plot(axes[1], "idd", conf, f"(b) IDD (ITA, {IDD_SPLIT}) -- conf={cs(conf)}", legend=False)
        fig.tight_layout()
        fname = f"fig_combined_c{cs(conf).replace('.','')}.pdf"
        fig.savefig(base / fname, bbox_inches="tight"); plt.close(fig)
        print(f"  Figure -> {base / fname}")

    # per-class recall vs IoU at the operating point conf=0.25 (recall-per-mine)
    # 2 panels (ODD | IDD); pfm1 = solid, starfish = dashed; one colour per model.
    c_op = cs(0.25)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, dom, dtitle in [(axes[0], "odd", f"(a) ODD (USA, {ODD_SPLIT})"),
                            (axes[1], "idd", f"(b) IDD (ITA, {IDD_SPLIT})")]:
        ax.set_title(f"{dtitle} -- per-class recall vs. IoU (conf=0.25)", fontweight="bold")
        for m in fig_subset:
            for cls_name, lstyle in [("pfm1", "-"), ("starfish", "--")]:
                ys = [results[m][dom][c_op][t]["classes"][cls_name]["recall"] for t in IOU_SWEEP]
                ax.plot(IOU_SWEEP, ys, color=colors[m], linestyle=lstyle, linewidth=1.8,
                        marker=markers[m], markersize=4,
                        label=f"{CONFIG_LABELS.get(m, m)} ({cls_name})")
        ax.set_xlabel("IoU threshold"); ax.set_ylabel("Recall")
        ax.set_xlim(-0.01, 0.52); ax.set_ylim(0.0, 1.0)
        ax.set_xticks([0.001, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
        ax.set_xticklabels(["0.001","0.05","0.10","0.20","0.30","0.40","0.50"], rotation=30)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        if dom == "odd":
            ax.legend(loc="lower left", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(base / "fig_perclass_recall_vs_iou.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"  Figure -> {base / 'fig_perclass_recall_vs_iou.pdf'}")

    # per-class recall grouped bar at the operating point (conf=0.25, IoU=REF_IOU)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, dom, dtitle in [(axes[0], "odd", f"ODD (USA, {ODD_SPLIT})"),
                            (axes[1], "idd", f"IDD (ITA, {IDD_SPLIT})")]:
        labels = [CONFIG_LABELS.get(m, m) for m in fig_subset]
        pfm1_r = [results[m][dom][c_op][REF_IOU]["classes"]["pfm1"]["recall"] for m in fig_subset]
        sfsh_r = [results[m][dom][c_op][REF_IOU]["classes"]["starfish"]["recall"] for m in fig_subset]
        x = np.arange(len(fig_subset)); bw = 0.38
        ax.bar(x - bw/2, pfm1_r, bw, label="pfm1",     color="#2c5f8e", alpha=0.85)
        ax.bar(x + bw/2, sfsh_r, bw, label="starfish", color="#e67e22", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Recall"); ax.set_ylim(0, 1.05)
        ax.set_title(f"{dtitle} -- recall by class (conf=0.25, IoU={REF_IOU})", fontweight="bold")
        ax.legend(); ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    fig.savefig(base / "fig_perclass_recall_bar.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"  Figure -> {base / 'fig_perclass_recall_bar.pdf'}")

    # confidence degradation: MF1 at REF_IOU vs conf
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, dom, dtitle in [(axes[0], "odd", f"ODD (USA, {ODD_SPLIT})"),
                            (axes[1], "idd", f"IDD (ITA, {IDD_SPLIT})")]:
        ax.set_title(f"{dtitle} -- Macro-F1 vs. confidence (IoU={REF_IOU})", fontweight="bold")
        for m in fig_subset:
            ys = [results[m][dom][cs(c)][REF_IOU]["macro_f1"] for c in CONF_LEVELS]
            ax.plot(CONF_LEVELS, ys, color=colors[m], linestyle=lss[m], linewidth=LW,
                    marker=markers[m], markersize=MS, label=CONFIG_LABELS.get(m, m))
        ax.set_xlabel("Confidence threshold"); ax.set_ylabel("Macro-F1")
        ax.set_xticks(CONF_LEVELS); ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        if dom == "odd":
            ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(base / "fig_conf_degradation.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"  Figure -> {base / 'fig_conf_degradation.pdf'}")


if __name__ == "__main__":
    main()
