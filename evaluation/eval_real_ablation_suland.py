#!/usr/bin/env python3
"""
SSRR 2026 - Real yolov11L SULAND evaluation.

Evaluates the real model (trained on IDD-train, val on IDD-val, test on IDD-test)
on the same splits as all synthetic models:
  IDD : ITA.yolo / test      (held-out; the real model SELECTED its best.pt on
                              ITA-val, so evaluating on val would over-state its
                              IDD score. test is unseen by every model -> fair.)
  ODD : USA.yolo / val       (USA has no train/test; val IS the OOD test set)

Outputs (appended into the latest ablation run dir, or --outdir):
  per_model/real_vv.json
  real_summary.csv

Usage:
    python eval_real_ablation_suland.py
    python eval_real_ablation_suland.py --outdir eval_results/ablation_suland/run_YYYYMMDD
    python eval_real_ablation_suland.py --resume
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
_RUN_SEED    = int(os.environ.get("SSRR_RUN_SEED", 42))
_SEED_SUFFIX = "" if _RUN_SEED == 42 else f"_seed{_RUN_SEED}"   # resolve _seed{N} weights for alt seeds
REAL_WEIGHTS = (SSRR_ROOT / "runs" / "detect" / "real_ablation"
                / f"real_idd_yolo11l_adamw_100ep{_SEED_SUFFIX}" / "best.pt")
RESULTS_BASE = SSRR_ROOT / "eval_results" / "ablation_suland"

# IDD on held-out TEST (fair to the real model, which selected best.pt on val).
# ODD on USA-val (the only USA split = OOD test set). Must match eval_ablation_suland.py.
IDD_SPLIT  = "test"
ODD_SPLIT  = "val"
IDD_IMAGES = SULAND_ROOT / "iid" / "ITA.yolo" / IDD_SPLIT / "images"
IDD_LABELS = SULAND_ROOT / "iid" / "ITA.yolo" / IDD_SPLIT / "labels"
ODD_IMAGES = SULAND_ROOT / "ood" / "USA.yolo" / ODD_SPLIT / "images"
ODD_LABELS = SULAND_ROOT / "ood" / "USA.yolo" / ODD_SPLIT / "labels"
IDD_TAG    = f"IDD-{IDD_SPLIT}"
ODD_TAG    = f"ODD-{ODD_SPLIT}"

# ---------------------------------------------------------------------------
# Constants (identical to eval_ablation_suland.py)
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

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Helpers (identical to eval_ablation_suland.py)
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


def gt_from_cache(img_path, cache, shape):
    h, w = shape
    return [(cls, (xc-nw/2)*w, (yc-nh/2)*h, (xc+nw/2)*w, (yc+nh/2)*h)
            for cls, xc, yc, nw, nh in cache.get(img_path, [])]


def update_matrix(matrix, gt_boxes, pred_result):
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
    xl = np.maximum(gboxes[:,0:1], pboxes[:,0]); yt = np.maximum(gboxes[:,1:2], pboxes[:,1])
    xr = np.minimum(gboxes[:,2:3], pboxes[:,2]); yb = np.minimum(gboxes[:,3:4], pboxes[:,3])
    inter   = np.maximum(0.0, xr-xl) * np.maximum(0.0, yb-yt)
    ag      = (gboxes[:,2]-gboxes[:,0]) * (gboxes[:,3]-gboxes[:,1])
    ap      = (pboxes[:,2]-pboxes[:,0]) * (pboxes[:,3]-pboxes[:,1])
    iou_mat = np.where((ag[:,None]+ap[None,:]-inter) > 0,
                       inter/(ag[:,None]+ap[None,:]-inter), 0.0)
    used = np.zeros(P, dtype=bool)
    for g in range(G):
        row = iou_mat[g].copy(); row[used] = -1.0
        bi  = int(np.argmax(row))
        if row[bi] >= IOU_THRESHOLD:
            matrix[gcls[g], pcls[bi]] += 1; used[bi] = True
        else:
            matrix[gcls[g], NC] += 1
    for pi in range(P):
        if not used[pi]:
            matrix[NC, pcls[pi]] += 1


def matrix_metrics(matrix):
    classes = {}
    f1s, precs, recs = [], [], []
    for i, name in enumerate(CLASS_NAMES):
        tp = matrix[i,i]
        fp = matrix[:NC,i].sum() - tp + matrix[NC,i]
        fn = matrix[i,:NC].sum() - tp + matrix[i,NC]
        support = matrix[i,:].sum()
        prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
        rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        classes[name] = {"tp":int(tp),"fp":int(fp),"fn":int(fn),"support":int(support),
                         "precision":round(prec,6),"recall":round(rec,6),"f1":round(f1,6)}
        f1s.append(f1); precs.append(prec); recs.append(rec)
    supports = [classes[n]["support"] for n in CLASS_NAMES]
    total_s  = sum(supports) or 1
    return {"classes": classes,
            "macro_f1":       round(float(np.mean(f1s)),6),
            "macro_precision":round(float(np.mean(precs)),6),
            "macro_recall":   round(float(np.mean(recs)),6),
            "weighted_f1":    round(sum(f1s[i]*supports[i] for i in range(NC))/total_s,6),
            "total_gt":       int(sum(supports))}


def run_inference(model, images, gt_cache, conf, tag):
    matrix = np.zeros((NC+1, NC+1), dtype=np.float64)
    total  = len(images)
    print(f"\n  -- {tag} --  ({total} images, conf={conf})")
    for start in range(0, total, BATCH_SIZE):
        batch   = images[start:start+BATCH_SIZE]
        results = model.predict(source=batch, conf=conf, batch=len(batch),
                                device=DEVICE, half=HALF, imgsz=IMGSZ, verbose=False)
        for img_path, pred in zip(batch, results):
            update_matrix(matrix, gt_from_cache(img_path, gt_cache, pred.orig_shape), pred)
        done = min(start+len(batch), total)
        m  = matrix_metrics(matrix)
        print(f"\r  [{tag}] {done}/{total}  MacF1={m['macro_f1']:.3f}"
              f"  pfm1 Rec={m['classes']['pfm1']['recall']:.3f}"
              f"  sfsh Rec={m['classes']['starfish']['recall']:.3f}", end="", flush=True)
    print()
    return matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--size", default="l", choices=["l", "s", "n"],
                        help="YOLO11 model size to evaluate (default: l)")
    args = parser.parse_args()

    # size-specific weights + output names (n/s must not clobber the l outputs)
    real_dir = (SSRR_ROOT / "runs" / "detect" / "real_ablation"
                / f"real_idd_yolo11{args.size}_adamw_100ep{_SEED_SUFFIX}")
    # L was copied as a flat best.pt; freshly-trained n/s live under weights/
    real_weights = (real_dir / "weights" / "best.pt"
                    if (real_dir / "weights" / "best.pt").exists() else real_dir / "best.pt")
    suffix    = "" if args.size == "l" else f"_{args.size}"
    cfg_name  = f"real_vv{suffix}"

    if args.outdir:
        base = Path(args.outdir)
    else:
        runs = sorted(RESULTS_BASE.glob("run_*"))
        if not runs:
            print(f"[ERROR] No ablation runs in {RESULTS_BASE}. Run eval_ablation_suland.py first.")
            return
        base = runs[-1]

    base.mkdir(parents=True, exist_ok=True)
    per_model_dir = base / "per_model"
    per_model_dir.mkdir(exist_ok=True)
    json_path = per_model_dir / f"{cfg_name}.json"
    print(f"\n  Output dir : {base}   (yolo11{args.size})")

    if not real_weights.exists():
        print(f"[ERROR] Weights not found: {real_weights}"); return

    if args.resume and json_path.exists():
        print(f"  [RESUME] Loading cached {cfg_name} results")
        result = json.loads(json_path.read_text())
    else:
        for name, path in [(f"{IDD_TAG} images", IDD_IMAGES), (f"{IDD_TAG} labels", IDD_LABELS),
                           (f"{ODD_TAG} images", ODD_IMAGES), (f"{ODD_TAG} labels", ODD_LABELS)]:
            if not path.exists():
                print(f"[ERROR] {name} not found: {path}"); return

        idd_images = get_image_files(IDD_IMAGES)
        odd_images = get_image_files(ODD_IMAGES)
        print(f"  {IDD_TAG} : {len(idd_images)} images  |  {ODD_TAG} : {len(odd_images)} images\n")

        print("  Pre-loading GT caches...", flush=True)
        idd_gt = preload_gt_cache(idd_images, IDD_IMAGES, IDD_LABELS)
        odd_gt = preload_gt_cache(odd_images, ODD_IMAGES, ODD_LABELS, ODD_CLASS_REMAP)

        print(f"\n  Loading real model: {real_weights}", flush=True)
        model  = YOLO(str(real_weights))
        result = {}
        t_total = time.time()

        for conf in CONF_THRESHOLDS:
            cs      = f"{conf:.2f}"
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
            print(f"  conf={conf}  IDD ({t1-t0:.0f}s) MacF1={im['macro_f1']:.3f}"
                  f"  pfm1 Rec={im['classes']['pfm1']['recall']:.3f}"
                  f"  sfsh Rec={im['classes']['starfish']['recall']:.3f}", flush=True)
            print(f"           ODD ({t2-t1:.0f}s) MacF1={om['macro_f1']:.3f}"
                  f"  pfm1 Rec={om['classes']['pfm1']['recall']:.3f}"
                  f"  sfsh Rec={om['classes']['starfish']['recall']:.3f}", flush=True)

        print(f"\n  Total: {(time.time()-t_total)/60:.1f} min")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        json_path.write_text(json.dumps(result, indent=2))
        print(f"  Saved -> {json_path}")

    # Write real_summary.csv
    fieldnames = [
        "config", "conf", "domain", "n_imgs",
        "macro_f1", "macro_precision", "macro_recall", "weighted_f1",
        "pfm1_tp","pfm1_fp","pfm1_fn","pfm1_support","pfm1_precision","pfm1_recall","pfm1_f1",
        "starfish_tp","starfish_fp","starfish_fn","starfish_support",
        "starfish_precision","starfish_recall","starfish_f1",
    ]
    rows = []
    for conf in CONF_THRESHOLDS:
        cs = f"{conf:.2f}"
        for dom in ["idd", "odd"]:
            r  = result[cs][dom]
            m  = r["metrics"]
            p1 = m["classes"]["pfm1"]
            sf = m["classes"]["starfish"]
            rows.append({
                "config": cfg_name, "conf": conf, "domain": dom, "n_imgs": r["n_imgs"],
                "macro_f1": m["macro_f1"], "macro_precision": m["macro_precision"],
                "macro_recall": m["macro_recall"], "weighted_f1": m["weighted_f1"],
                "pfm1_tp": p1["tp"], "pfm1_fp": p1["fp"], "pfm1_fn": p1["fn"],
                "pfm1_support": p1["support"], "pfm1_precision": p1["precision"],
                "pfm1_recall": p1["recall"], "pfm1_f1": p1["f1"],
                "starfish_tp": sf["tp"], "starfish_fp": sf["fp"], "starfish_fn": sf["fn"],
                "starfish_support": sf["support"], "starfish_precision": sf["precision"],
                "starfish_recall": sf["recall"], "starfish_f1": sf["f1"],
            })

    csv_path = base / f"real_summary{suffix}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV  -> {csv_path}")

    print(f"\n  DONE\n")


if __name__ == "__main__":
    main()
