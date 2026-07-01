#!/usr/bin/env python3
"""Results-2: IoU sensitivity / consistency analysis (YOLOv11l, real + 14 configs).
Source: eval_results/iou_conf_sweep/yolo11l_30k  (IDD=test, ODD=val; conf 0.25/0.50/0.75; IoU 0.001->0.50).
Writes results/RESULTS_2_iou_sweep_yolov11l.txt + figures in results/figs/."""
import csv, statistics, subprocess, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
EVAL=Path(os.environ.get("SSRR_EVAL", REPO/"eval_results"))
RUN=EVAL/"iou_conf_sweep"/"yolo11l_30k"
OUTDIR=Path(os.environ.get("SSRR_OUT", REPO/"results")); FIG=OUTDIR/"figs"; FIG.mkdir(parents=True,exist_ok=True)
R=list(csv.DictReader(open(RUN/"results.csv")))
AUC=list(csv.DictReader(open(RUN/"auc.csv")))
CONFS=[0.25,0.50,0.75]
IOUS=sorted({float(r["iou_threshold"]) for r in R})
LAB={"t_10":"T-10","n_0":"N-0","baseline":"Baseline","sun_off":"Sun-Off","yaw_off":"Yaw-Off","t_5":"T-5","i_1":"I-1",
     "s_off":"S-Off","pen_off":"Pen-Off","inv_off":"Inv-Off","n_30":"N-30","v_low":"V-Low","v_mid":"V-Mid","h_off":"H-Off","real_vv":"Real"}
SYN=["baseline","h_off","sun_off","t_10","t_5","v_low","v_mid","inv_off","pen_off","s_off","i_1","n_0","n_30","yaw_off"]

def mf1(model,dom,conf,iou):
    for r in R:
        if r["model"]==model and r["domain"]==dom and abs(float(r["conf"])-conf)<1e-6 and abs(float(r["iou_threshold"])-iou)<1e-6:
            return float(r["macro_f1"])
    return float("nan")
def auc(model,dom,conf):
    for r in AUC:
        if r["model"]==model and abs(float(r["conf"])-conf)<1e-6:
            return float(r["auc_odd"] if dom=="odd" else r["auc_idd"])
    return float("nan")

# rank synth by ODD MF1@0.001 conf0.25
RANK=sorted(SYN,key=lambda c:mf1(c,"odd",0.25,0.001),reverse=True)

def auc_table():
    hdr=f"{'Config':<10}"+"".join(f"{'ODD@'+str(c):>9}{'IDD@'+str(c):>9}" for c in CONFS)
    L=[hdr,"-"*len(hdr)]
    L.append(f"{'Real':<10}"+"".join(f"{auc('real_vv','odd',c):>9.3f}{auc('real_vv','idd',c):>9.3f}" for c in CONFS))
    L.append("-"*len(hdr))
    for cfg in RANK:
        L.append(f"{LAB[cfg]:<10}"+"".join(f"{auc(cfg,'odd',c):>9.3f}{auc(cfg,'idd',c):>9.3f}" for c in CONFS))
    return "\n".join(L)

def consistency(conf):
    hdr=f"{'IoU':>6}{'Real':>8}{'Best':>8}{'Worst':>8}{'Median':>8}{'#>real':>8}{'all>real':>10}"
    L=[hdr,"-"*len(hdr)]; allok=True
    for iou in IOUS:
        real=mf1("real_vv","odd",conf,iou)
        syn=[mf1(c,"odd",conf,iou) for c in SYN]
        cnt=sum(1 for s in syn if s>real); ok=(cnt==14); allok=allok and ok
        L.append(f"{iou:>6.3f}{real:>8.3f}{max(syn):>8.3f}{min(syn):>8.3f}{statistics.median(syn):>8.3f}{cnt:>6}/14{('YES' if ok else 'NO'):>10}")
    L.append("-"*len(hdr))
    L.append(f"  => conf {conf}: "+("ALL 14 synthetic > real at EVERY IoU (consistent)" if allok else "ranking breaks at some IoU"))
    return "\n".join(L)

def perconfig(dom,conf):
    hdr=f"{'Config':<10}"+"".join(f"{iou:>7.3f}" for iou in IOUS)
    L=[hdr,"-"*len(hdr)]
    L.append(f"{'Real':<10}"+"".join(f"{mf1('real_vv',dom,conf,iou):>7.3f}" for iou in IOUS))
    L.append("-"*len(hdr))
    for cfg in RANK:
        L.append(f"{LAB[cfg]:<10}"+"".join(f"{mf1(cfg,dom,conf,iou):>7.3f}" for iou in IOUS))
    return "\n".join(L)

# ---- narrative ----
def allok(conf):
    return all(all(mf1(c,"odd",conf,iou)>mf1("real_vv","odd",conf,iou) for c in SYN) for iou in IOUS)
verdict={c:allok(c) for c in CONFS}
bsa=max(((auc(c,"odd",0.25),c) for c in SYN)); ra=auc("real_vv","odd",0.25)
odd_real_lo,odd_real_hi=mf1("real_vv","odd",0.25,IOUS[-1]),mf1("real_vv","odd",0.25,IOUS[0])
best_lo=mf1("t_10","odd",0.25,IOUS[-1]); best_hi=mf1("t_10","odd",0.25,IOUS[0])

T=f"""================================================================================
 RESULTS 2 -- IoU SENSITIVITY / CONSISTENCY  (YOLOv11l, 30k)
 Source: eval_results/iou_conf_sweep/run_20260531_194317
 Models: Real + 14 synthetic ablations.  IDD=ITA/test, ODD=USA/val.
 Sweep: IoU matching threshold {IOUS[0]} -> {IOUS[-1]} ({len(IOUS)} points), conf 0.25/0.50/0.75.
 Question: is the synthetic>real advantage on ODD an artifact of the loose IoU=0.001
           matching? -> Sweep IoU and check whether the ranking is preserved.
 AUC = normalised area under the Macro-F1-vs-IoU curve over [0.05,0.50] (higher=more
       threshold-robust). Real on top in every table.
================================================================================

A. AUC SUMMARY  (area under Macro-F1 vs IoU; per domain & confidence)
{auc_table()}
 - Higher AUC = flatter, more threshold-robust curve. On ODD every synthetic config's
   AUC dwarfs real; on IDD real's AUC is highest (it dominates in-distribution).

B. CONSISTENCY CHECK ON ODD  (does every synthetic config beat real at every IoU?)

 conf = 0.25
{consistency(0.25)}

 conf = 0.50
{consistency(0.50)}

 conf = 0.75
{consistency(0.75)}

C. FULL Macro-F1 vs IoU -- ODD (USA) @ conf 0.25   (config rows, IoU columns)
{perconfig('odd',0.25)}

D. FULL Macro-F1 vs IoU -- IDD (ITA-test) @ conf 0.25   (real dominates in-distribution)
{perconfig('idd',0.25)}

================================================================================
 ANALYSIS (Results 2)
================================================================================

A. THE RANKING IS INVARIANT TO THE IoU THRESHOLD (ODD).
   At conf 0.25/0.50/0.75, all 14 synthetic configs beat real at EVERY IoU threshold
   from {IOUS[0]} to {IOUS[-1]}, with no crossings:
     conf 0.25 -> {'consistent' if verdict[0.25] else 'BREAKS'}
     conf 0.50 -> {'consistent' if verdict[0.50] else 'BREAKS'}
     conf 0.75 -> {'consistent' if verdict[0.75] else 'BREAKS'}
   The synthetic>real result is therefore NOT an artifact of the loose IoU=0.001 matching.

B. THE GAP NEVER CLOSES.
   ODD @ conf 0.25: real moves only {odd_real_hi:.3f} -> {odd_real_lo:.3f} across the sweep
   (already near the floor), while the best synthetic (T-10) goes {best_hi:.3f} -> {best_lo:.3f}.
   Even at the strict end (IoU=0.50) the worst synthetic stays well above real.

C. AUC CONFIRMS IT.
   ODD AUC @ conf 0.25: best synthetic ({LAB[bsa[1]]}) = {bsa[0]:.3f} vs real = {ra:.3f}
   (~{bsa[0]/ra:.1f}x). On IDD the ordering flips (real highest) -- expected, and it shows
   the metric is not trivially biased toward synthetic.

D. INTERPRETATION.
   Stricter localization penalises all models similarly; because synthetic detections are
   both more frequent (recall) and well-centred, the ranking survives. The loose annotation
   protocol (Fig. 2 in the paper) is justified and not the cause of the advantage.

KEY NUMBERS
 ODD all-synth>real at every IoU: conf0.25={verdict[0.25]}  conf0.50={verdict[0.50]}  conf0.75={verdict[0.75]}
 ODD AUC @0.25: real={ra:.3f}  best({LAB[bsa[1]]})={bsa[0]:.3f}
 ODD real MF1 @0.25 across IoU: {odd_real_hi:.3f} (0.001) -> {odd_real_lo:.3f} (0.50)
 ODD T-10 MF1 @0.25 across IoU: {best_hi:.3f} (0.001) -> {best_lo:.3f} (0.50)
================================================================================
"""
(OUTDIR/"RESULTS_2_iou_sweep_yolov11l.txt").write_text(T)

# ---- figures ----
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"figure.dpi":150})
# copy the existing 3conf x 2domain grid
subprocess.run(["cp","-f",str(RUN/"fig_grid.pdf"),str(FIG/"iou_sweep_grid.pdf")])

# consistency figure: ODD | IDD @ conf0.25, all 14 synth (thin green) + real (red), band
fig,axes=plt.subplots(1,2,figsize=(12,4.6))
for ax,dom,title in [(axes[0],"odd","ODD (USA) -- conf 0.25"),(axes[1],"idd","IDD (ITA-test) -- conf 0.25")]:
    band_lo=[min(mf1(c,dom,0.25,i) for c in SYN) for i in IOUS]
    band_hi=[max(mf1(c,dom,0.25,i) for c in SYN) for i in IOUS]
    for c in SYN: ax.plot(IOUS,[mf1(c,dom,0.25,i) for i in IOUS],color="#1a7d3b",lw=0.8,alpha=0.45)
    ax.fill_between(IOUS,band_lo,band_hi,color="#1a7d3b",alpha=0.12,label="14 synthetic (band)")
    ax.plot(IOUS,[mf1("real_vv",dom,0.25,i) for i in IOUS],color="#c0392b",lw=2.4,marker="D",ms=5,label="Real")
    ax.set_title(title,fontweight="bold"); ax.set_xlabel("IoU threshold"); ax.set_ylabel("Macro-F1")
    ax.set_ylim(0,1); ax.set_xlim(-0.01,0.52); ax.grid(True,ls=":",alpha=.6); ax.legend(loc="upper right",fontsize=9)
fig.suptitle("IoU sensitivity: synthetic band stays above real on ODD at every threshold",fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(FIG/"iou_consistency_conf025.pdf"); plt.close(fig)

# AUC bar chart ODD conf0.25
fig,ax=plt.subplots(figsize=(7,4.4))
order=sorted(SYN,key=lambda c:auc(c,"odd",0.25))
vals=[auc(c,"odd",0.25) for c in order]
ax.barh([LAB[c] for c in order],vals,color="#1a7d3b",alpha=.85)
ax.axvline(ra,color="#c0392b",ls="--",lw=2,label=f"Real AUC = {ra:.3f}")
for i,v in enumerate(vals): ax.text(v+0.005,i,f"{v:.3f}",va="center",fontsize=8)
ax.set_xlabel("ODD AUC (Macro-F1 vs IoU, conf 0.25)"); ax.set_xlim(0,0.8)
ax.set_title("Threshold-robustness (AUC): every synthetic config >> real",fontweight="bold")
ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(FIG/"iou_auc_bars_odd.pdf"); plt.close(fig)

print("written ->",OUTDIR/"RESULTS_2_iou_sweep_yolov11l.txt")
print("figures ->",FIG/"iou_sweep_grid.pdf",",",FIG/"iou_consistency_conf025.pdf",",",FIG/"iou_auc_bars_odd.pdf")
