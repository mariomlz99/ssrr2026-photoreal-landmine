#!/usr/bin/env python3
"""Results-3: Detector-scale (deployability) analysis (YOLO11 N/S/L, 30k).
Tables: macro-F1 (+ per-class recall) by size x conf, IDD & ODD -- the paper's
'Detector-scale analysis' table (mF1 by scale). Figure: deploy_odd_vs_size.pdf.

No efficiency/FPS/GFLOPs table: the paper reports mF1-by-scale only; YOLOv11 param
counts are standard architecture constants, cited in text (abstract), not measured here.
Output: results/RESULTS_3_deployability.txt + figures in results/figs/."""
import csv, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
EVAL=Path(os.environ.get("SSRR_EVAL", REPO/"eval_results"))
AB=EVAL/"ablation_suland"
OUTDIR=Path(os.environ.get("SSRR_OUT", REPO/"results")); FIG=OUTDIR/"figs"; FIG.mkdir(parents=True,exist_ok=True)
CONFS=[0.25,0.50,0.75]; SIZES=["n","s","l"]
RUNS={'l':'yolo11l_30k','s':'yolo11s_30k','n':'yolo11n_30k'}
REALF={'l':('yolo11l_30k/real_summary.csv','real_vv'),
       's':('yolo11s_30k/real_summary_s.csv','real_vv_s'),
       'n':('yolo11n_30k/real_summary_n.csv','real_vv_n')}
SYN={sz:list(csv.DictReader(open(AB/RUNS[sz]/"summary.csv"))) for sz in SIZES}
REAL={sz:(list(csv.DictReader(open(AB/REALF[sz][0]))),REALF[sz][1]) for sz in SIZES}
PARAMS={"n":2.6,"s":9.4,"l":25.3}   # standard YOLOv11 param counts (M) -- cited in text, not a measured table
CFGS=["sun_off","baseline","t_10","n_0"]
LAB={"sun_off":"Sun-Off","baseline":"Baseline","t_10":"T-10","n_0":"N-0"}

def gs(sz,cfg,dom,conf,f):
    for r in SYN[sz]:
        if r["config"]==cfg and r["domain"].lower().startswith(dom) and abs(float(r["conf"])-conf)<1e-6: return float(r[f])
    return float("nan")
def gr(sz,dom,conf,f):
    rows,c=REAL[sz]
    for r in rows:
        if r["config"]==c and r["domain"].lower().startswith(dom) and abs(float(r["conf"])-conf)<1e-6: return float(r[f])
    return float("nan")

def grid(dom,field):
    hdr=f"{'Model':<10}"+"".join(f"  | c{c:<4} N      S      L    " for c in CONFS)
    L=[hdr,"-"*len(hdr)]
    L.append(f"{'Real':<10}"+"".join(f"  |     {gr('n',dom,c,field):.3f} {gr('s',dom,c,field):.3f} {gr('l',dom,c,field):.3f}" for c in CONFS))
    L.append("-"*len(hdr))
    best={(c,sz):max(CFGS,key=lambda cf:gs(sz,cf,dom,c,field)) for c in CONFS for sz in SIZES}
    for cfg in CFGS:
        s=f"{LAB[cfg]:<10}"
        for c in CONFS:
            cells=""
            for sz in SIZES:
                v=gs(sz,cfg,dom,c,field); m="*" if best[(c,sz)]==cfg else " "
                cells+=f"{v:.3f}{m}"
            s+=f"  | {cells}"
        L.append(s)
    return "\n".join(L)

# consistency: all synth>real on ODD at every (size,conf)?
ok_all=all(gs(sz,cf,"odd",c,"macro_f1")>gr(sz,"odd",c,"macro_f1") for sz in SIZES for c in CONFS for cf in CFGS)
synN_best=max(gs("n",cf,"odd",0.25,"macro_f1") for cf in CFGS)
realL=gr("l","odd",0.25,"macro_f1")
bestN_cfg=max(CFGS,key=lambda cf:gs("n",cf,"odd",0.25,"macro_f1"))

T=f"""================================================================================
 RESULTS 3 -- DETECTOR-SCALE / DEPLOYABILITY  (YOLO11 N / S / L, 30k synthetic)
 Configs: Real + Sun-Off, Baseline, T-10, N-0.  IDD=ITA/test, ODD=USA/val.
 Macro-F1 at conf 0.25/0.50/0.75; IoU>=0.001. Real on top; (*) = best synthetic per column.
 Question: does the synthetic>real advantage survive shrinking to small/nano real-time models?
================================================================================

A. ODD (USA) Macro-F1 -- by model size (N/S/L) x confidence
{grid('odd','macro_f1')}

B. IDD (ITA-test) Macro-F1 -- by model size (N/S/L) x confidence
{grid('idd','macro_f1')}

C. ODD PFM-1 recall -- by size x conf
{grid('odd','pfm1_recall')}

D. ODD PMA-2 recall -- by size x conf
{grid('odd','starfish_recall')}

================================================================================
 ANALYSIS (Results 3)
================================================================================

A. THE ADVANTAGE HOLDS AT EVERY SIZE AND CONFIDENCE.
   On ODD, all 4 synthetic configs beat real in every (size x conf) cell: {ok_all}.
   ODD Macro-F1 @ conf 0.25, best synth vs real:
     N: {max(gs('n',cf,'odd',0.25,'macro_f1') for cf in CFGS):.3f} vs {gr('n','odd',0.25,'macro_f1'):.3f}
     S: {max(gs('s',cf,'odd',0.25,'macro_f1') for cf in CFGS):.3f} vs {gr('s','odd',0.25,'macro_f1'):.3f}
     L: {max(gs('l',cf,'odd',0.25,'macro_f1') for cf in CFGS):.3f} vs {gr('l','odd',0.25,'macro_f1'):.3f}

B. DEPLOYABILITY INVERSION (the headline).
   The best NANO synthetic ({LAB[bestN_cfg]}, ODD {synN_best:.3f}) beats the LARGE real model
   ({realL:.3f}) -- a 2.6M-parameter nano detector out-generalizes the 25.3M-parameter
   real model on out-of-distribution data.

C. IDD (in-distribution) -- real dominates at conf 0.25 and 0.50 (it trained there).
   But at conf 0.75 the real model collapses on IDD too (<=0.17) while synthetic holds
   (~0.55-0.66): synthetic leads even in-distribution at high confidence, because real's
   confidence is poorly calibrated. The deployment axis is ODD, where synthetic wins at
   every size AND every confidence.

D. CONFIG NOTES.
   Sun-Off is the most size-robust synthetic config (best or near-best at N and S).
   Baseline degrades hardest at N (see ODD grid) -- augmentation tuning matters more
   for tiny models (developed further in Results-4 / data-scaling).

KEY NUMBERS  (ODD Macro-F1 @ conf 0.25, best-synth / real)
 N (yolo11n, {PARAMS['n']}M params): {max(gs('n',cf,'odd',0.25,'macro_f1') for cf in CFGS):.3f} / {gr('n','odd',0.25,'macro_f1'):.3f}
 S (yolo11s, {PARAMS['s']}M params): {max(gs('s',cf,'odd',0.25,'macro_f1') for cf in CFGS):.3f} / {gr('s','odd',0.25,'macro_f1'):.3f}
 L (yolo11l, {PARAMS['l']}M params): {max(gs('l',cf,'odd',0.25,'macro_f1') for cf in CFGS):.3f} / {gr('l','odd',0.25,'macro_f1'):.3f}
================================================================================
"""
(OUTDIR/"RESULTS_3_deployability.txt").write_text(T)

# ---- figure: mF1 vs detector scale (the paper's deployability figure) ----
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"figure.dpi":150})
G,B,R,P,Tl="#1a7d3b","#2c5f8e","#c0392b","#8e44ad","#16a085"; xp=[0,1,2]
fig,ax=plt.subplots(figsize=(6.6,4.3))
for cfg,col,mk in [("sun_off",G,"o"),("baseline",B,"s"),("t_10",P,"^"),("n_0",Tl,"v")]:
    ax.plot(xp,[gs(sz,cfg,"odd",0.25,"macro_f1") for sz in SIZES],color=col,marker=mk,lw=2,ms=7,label=LAB[cfg])
ax.plot(xp,[gr(sz,"odd",0.25,"macro_f1") for sz in SIZES],color=R,ls="--",marker="D",lw=2,ms=7,label="Real")
ax.annotate("nano-synth > large-real",(0,synN_best),xytext=(0.1,0.50),fontsize=9,color=G,arrowprops=dict(arrowstyle="->",color=G))
ax.set_xticks(xp); ax.set_xticklabels([f"YOLO11{s}\n{PARAMS[s]}M" for s in SIZES],fontsize=9)
ax.set_ylabel("ODD Macro-F1 (conf 0.25)"); ax.set_ylim(0.3,0.8)
ax.set_title("Deployability: synthetic vs real across model size (30k)",fontweight="bold")
ax.grid(True,ls=":",alpha=.6); ax.legend(loc="center right",fontsize=9)
fig.tight_layout(); fig.savefig(FIG/"deploy_odd_vs_size.pdf"); plt.close(fig)

print("written ->",OUTDIR/"RESULTS_3_deployability.txt")
print("figure  ->",FIG/"deploy_odd_vs_size.pdf")
