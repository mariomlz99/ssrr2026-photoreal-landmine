#!/usr/bin/env python3
"""
SSRR 2026 -- Section IV-E evaluation: the six class-balance / volume control models
(iddbal & volctrl, each l/s/n) on the SULAND benchmark.

  IDD : ITA.yolo / test   (held-out)        ODD : USA.yolo / val   (OOD test)
  macro-F1 + per-class recall (PFM-1, PMA-2), conf 0.25/0.50/0.75, IoU >= 0.001,
  ODD class remap {0:1, 1:0}. Reuses the metric machinery of eval_ablation_suland.

Reads weights from  $SSRR_RUNS_BASE/runs/detect/synthetic_ablation/<run>/weights/best.pt
Writes  $SSRR_RUNS_BASE/eval_results/prevalence_volume/{summary.csv, full_results.txt}

Usage:
    ./eval_prevalence_volume.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `import eval_ablation_suland` works
import eval_ablation_suland as E

RUNS = E.RUNS_DIR                                            # runs/detect/synthetic_ablation
OUT  = E.SSRR_ROOT / "eval_results" / "prevalence_volume"
OUT.mkdir(parents=True, exist_ok=True)

# (label, prevalence, mines-rel-to-real, scale) -> run dir
MODELS = []
for size in ["n", "s", "l"]:
    MODELS.append(("iddbal",  "23%", size, f"baseline_iddbal_30k_yolo11{size}_adamw_100ep"))
for size in ["n", "s", "l"]:
    MODELS.append(("volctrl", "85%", size, f"baseline_volctrl_5304_yolo11{size}_adamw_100ep"))


def main():
    idd = E.get_image_files(E.IDD_IMAGES)
    odd = E.get_image_files(E.ODD_IMAGES)
    ig  = E.preload_gt_cache(idd, E.IDD_IMAGES, E.IDD_LABELS)
    og  = E.preload_gt_cache(odd, E.ODD_IMAGES, E.ODD_LABELS, E.ODD_CLASS_REMAP)

    rows = []
    for label, prev, size, rundir in MODELS:
        wp = RUNS / rundir / "weights" / "best.pt"
        if not wp.exists():
            print(f"  [SKIP] {rundir}: weights not found"); continue
        res = E.evaluate_model(f"{label}_{size}", wp, idd, ig, odd, og)
        for conf in E.CONF_THRESHOLDS:
            cs = f"{conf:.2f}"
            for dom in ["idd", "odd"]:
                m = res[cs][dom]["metrics"]
                rows.append({
                    "set": label, "scale": f"yolov11{size}", "prevalence": prev,
                    "domain": dom.upper(), "conf": conf,
                    "macro_f1":    m["macro_f1"],
                    "recall_pfm1": m["classes"]["pfm1"]["recall"],
                    "recall_pma2": m["classes"]["starfish"]["recall"],
                })

    if not rows:
        print("[prevalence/volume] no models evaluated (train them first)."); return

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # paste-ready table
    lines = ["", "================ Sec IV-E: class-balance / volume control ================",
             "cols = macro-F1 / recall PFM-1 / recall PMA-2 ; IoU>=0.001", ""]
    order = [("iddbal", s) for s in ["n", "s", "l"]] + [("volctrl", s) for s in ["n", "s", "l"]]

    def get(setl, size, dom, conf, k):
        for r in rows:
            if (r["set"] == setl and r["scale"] == f"yolov11{size}"
                    and r["domain"] == dom and abs(r["conf"] - conf) < 1e-9):
                return r[k]
        return float("nan")

    for dom in ["IDD", "ODD"]:
        lines.append(f"---- {dom} ----")
        lines.append(f"{'model':<14}{'prev':<6}" +
                     "".join(f"{'mF1@'+str(c):>9}{'Rpfm':>7}{'Rpma':>7}" for c in E.CONF_THRESHOLDS))
        for setl, size in order:
            line = f"{setl+'_'+size:<14}{('23%' if setl=='iddbal' else '85%'):<6}"
            for c in E.CONF_THRESHOLDS:
                line += (f"{get(setl,size,dom,c,'macro_f1'):>9.3f}"
                         f"{get(setl,size,dom,c,'recall_pfm1'):>7.3f}"
                         f"{get(setl,size,dom,c,'recall_pma2'):>7.3f}")
            lines.append(line)
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (OUT / "full_results.txt").write_text(text + f"\nCSV -> {OUT/'summary.csv'}\n")
    print(f"CSV -> {OUT/'summary.csv'}")


if __name__ == "__main__":
    main()
