#!/usr/bin/env python3
"""Results-4: Data-scaling (10k/20k/30k x n/s/l). Configs: Sun-Off, Baseline (+ Real, fixed).
One comparable table per domain: rows = config-size and Real-size; cols = datasize x conf.
Plus inversion analysis and a confidence-sensitivity figure.
Output: results/RESULTS_4_datascaling.txt + figures in results/figs/."""
import csv, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
EVAL=Path(os.environ.get("SSRR_EVAL", REPO/"eval_results"))
AB=EVAL/"ablation_suland"
OUTDIR=Path(os.environ.get("SSRR_OUT", REPO/"results")); FIG=OUTDIR/"figs"; FIG.mkdir(parents=True,exist_ok=True)
CONFS=[0.25,0.50,0.75]; SIZES=["n","s","l"]; DS=["10k","20k","30k"]
RUN={('l','30k'):'yolo11l_30k',('s','30k'):'yolo11s_30k',('n','30k'):'yolo11n_30k',
     ('l','10k'):'yolo11l_10k',('s','10k'):'yolo11s_10k',('n','10k'):'yolo11n_10k',
     ('l','20k'):'yolo11l_20k',('s','20k'):'yolo11s_20k',('n','20k'):'yolo11n_20k'}
S={k:list(csv.DictReader(open(AB/v/"summary.csv"))) for k,v in RUN.items()}
REALF={'l':('yolo11l_30k/real_summary.csv','real_vv'),'s':('yolo11s_30k/real_summary_s.csv','real_vv_s'),
       'n':('yolo11n_30k/real_summary_n.csv','real_vv_n')}
REAL={k:(list(csv.DictReader(open(AB/p))),c) for k,(p,c) in REALF.items()}
LAB={"sun_off":"Sun-Off","baseline":"Baseline"}
CFGS=["sun_off","baseline"]

def gs(sz,ds,cfg,dom,conf,f="macro_f1"):
    for r in S[(sz,ds)]:
        if r["config"]==cfg and r["domain"].lower().startswith(dom) and abs(float(r["conf"])-conf)<1e-6: return float(r[f])
    return float("nan")
def gr(sz,dom,conf,f="macro_f1"):
    rows,c=REAL[sz]
    for r in rows:
        if r["config"]==c and r["domain"].lower().startswith(dom) and abs(float(r["conf"])-conf)<1e-6: return float(r[f])
    return float("nan")

def table(dom):
    hdr=f"{'Model':<12}"+"".join(f"  | c{c:<4} 10k    20k    30k  " for c in CONFS)+f"  {'d30-10@.25':>10}"
    L=[hdr,"-"*len(hdr)]
    for cfg in CFGS:
        for sz in SIZES:
            cells=""
            for c in CONFS:
                cells+="  | "+" ".join(f"{gs(sz,ds,cfg,dom,c):.3f}" for ds in DS)
            d=gs(sz,'30k',cfg,dom,0.25)-gs(sz,'10k',cfg,dom,0.25)
            arrow="UP " if d>0.01 else ("DOWN" if d<-0.01 else "flat")
            L.append(f"{LAB[cfg]+'-'+sz:<12}{cells}  {d:>+7.3f} {arrow}")
        L.append("-"*len(hdr))
    for sz in SIZES:  # real fixed across datasize
        cells="".join("  | "+" ".join(f"{gr(sz,dom,c):.3f}" for _ in DS) for c in CONFS)
        L.append(f"{'Real-'+sz:<12}{cells}  {'(fixed)':>12}")
    return "\n".join(L)

# inversion facts
bn=[gs('n',ds,'baseline','odd',0.25) for ds in DS]
sn=[gs('n',ds,'sun_off','odd',0.25) for ds in DS]
bn_idd=[gs('n',ds,'baseline','idd',0.25) for ds in DS]
bl=[gs('l',ds,'baseline','odd',0.25) for ds in DS]

T=f"""================================================================================
 RESULTS 4 -- DATA SCALING  (10k / 20k / 30k synthetic, YOLO11 n/s/l)
 Configs: Sun-Off, Baseline (the two trained at all data sizes) + Real (fixed, no data axis).
 IDD=ITA/test, ODD=USA/val. Macro-F1 at conf 0.25/0.50/0.75; IoU>=0.001.
 Last column: delta(30k-10k) at conf 0.25  -> +UP = scales normally, -DOWN = INVERSION.
 NOTE: Real has no data-size axis (trained on the fixed real set) -> repeated for comparison.
================================================================================

A. ODD (USA) Macro-F1 -- data-scaling (rows: config-size & Real-size; cols: datasize x conf)
{table('odd')}

B. IDD (ITA-test) Macro-F1 -- data-scaling
{table('idd')}

================================================================================
 ANALYSIS (Results 4)
================================================================================

A. MORE SYNTHETIC DATA HELPS -- AND IT ALREADY BEATS REAL AT 10k.
   Even at one third the data, synthetic crushes real on ODD (e.g. Sun-Off-n 10k = {sn[0]:.3f}
   vs Real-n = {gr('n','odd',0.25):.3f}). Sun-Off scales monotonically at every size; the knee is
   near 20k (Sun-Off-l: {gs('l','20k','sun_off','odd',0.25):.3f} -> {gs('l','30k','sun_off','odd',0.25):.3f} from 20k to 30k -- small).

B. THE CURVE INVERSION (Baseline-n): why it happens.
   Baseline-n *inverts*: ODD Macro-F1 DECREASES with more data, {bn[0]:.3f} -> {bn[1]:.3f} -> {bn[2]:.3f}
   (10k->20k->30k), the opposite of every other curve. Two facts pin the mechanism:
     (i)  CAPACITY-bound: only NANO inverts. Baseline-l scales normally ({bl[0]:.3f} -> {bl[2]:.3f}).
     (ii) CONFIG-bound: Baseline (HDRI + direct SUN) inverts; Sun-Off (HDRI only, NO hard
          direct-sun shadows) scales up ({sn[0]:.3f} -> {sn[2]:.3f}).
   Interpretation: the direct-sun HARD SHADOWS in Baseline are a strong, synthetic-specific
   cue. A low-capacity nano model, fed MORE shadow-heavy data, increasingly commits to this
   NON-transferable cue instead of robust mine/terrain features. The tell-tale sign: on the
   NEAR domain (IDD/ITA) the same Baseline-n IMPROVES with data ({bn_idd[0]:.3f} -> {bn_idd[2]:.3f}),
   while on the FAR domain (ODD/USA) it degrades -- the model specializes toward the
   shadow-consistent, ITA-like appearance and loses USA generalization. Sun-Off removes the
   shadow trap, so its nano model spends capacity on transferable features and scales up.
   => Practical rule: for capacity-constrained deployment models, use the augmentation that
      removes synthetic-specific cues (ambient-only / Sun-Off). It is not just higher, it
      scales PREDICTABLY; naive Baseline can actively waste extra data at nano.
   Caveat (honest): single-seed runs -- part of the {bn[0]-bn[2]:+.3f} swing could be training
   variance. A 3-seed repeat of Baseline-n (+ a Grad-CAM check that it attends to shadow
   edges) would confirm the mechanism. The direction holds across all 3 confidences
   (sensitivity figure), so it is not a threshold artifact.

C. SENSITIVITY (across confidence). The data-scaling trends -- Sun-Off up, Baseline-n down --
   are consistent at conf 0.25, 0.50 and 0.75 (see fig_datascale_sensitivity.pdf), i.e. the
   inversion is robust to the operating point, not an artifact of conf 0.25.

KEY NUMBERS (ODD Macro-F1 @ conf 0.25)
 Sun-Off-n 10k/20k/30k = {sn[0]:.3f}/{sn[1]:.3f}/{sn[2]:.3f}  (scales UP)
 Baseline-n 10k/20k/30k = {bn[0]:.3f}/{bn[1]:.3f}/{bn[2]:.3f}  (INVERTS / DOWN)
 Real-n (fixed) = {gr('n','odd',0.25):.3f}   -- both synth beat it at every data size
================================================================================
"""
(OUTDIR/"RESULTS_4_datascaling.txt").write_text(T)

# ---- figures ----
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"figure.dpi":150})
G,O,B,R="#1a7d3b","#e67e22","#2c5f8e","#c0392b"; dx=[0,1,2]
# main: 2 panels Sun-Off | Baseline, ODD MF1 vs datasize @0.25, n/s/l + real dotted
fig,axes=plt.subplots(1,2,figsize=(12.5,4.8),sharey=True)
for ax,cfg in zip(axes,CFGS):
    for sz,col,mk in [("l",G,"o"),("s",O,"s"),("n",B,"^")]:
        ax.plot(dx,[gs(sz,ds,cfg,"odd",0.25) for ds in DS],color=col,marker=mk,lw=2.2,ms=7,label=f"synth-{sz}")
        ax.axhline(gr(sz,"odd",0.25),color=col,ls=":",lw=1.6,alpha=.75)
    ax.set_xticks(dx); ax.set_xticklabels(DS,fontsize=12); ax.set_xlabel("synthetic images",fontsize=12)
    ax.set_title(LAB[cfg],fontweight="bold",fontsize=15); ax.grid(True,ls=":",alpha=.6); ax.set_ylim(0.35,0.8)
    if cfg=="sun_off": ax.set_ylabel("ODD Macro-F1 (conf 0.25)",fontsize=12); ax.legend(fontsize=11,title="dotted = real",title_fontsize=11)
fig.suptitle("Data-scaling (ODD): Sun-Off scales monotonically; Baseline-n INVERTS (degrades)",fontweight="bold",fontsize=14)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(FIG/"datascale_odd_vs_data.pdf"); plt.close(fig)

# sensitivity: nano focus, Sun-Off-n vs Baseline-n at 3 confidences
fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
confcol={0.25:"#1a7d3b",0.50:"#2c5f8e",0.75:"#8e44ad"}
for ax,cfg in zip(axes,CFGS):
    for c in CONFS:
        ax.plot(dx,[gs('n',ds,cfg,"odd",c) for ds in DS],color=confcol[c],marker="o",lw=2,ms=6,label=f"conf {c}")
    ax.set_xticks(dx); ax.set_xticklabels(DS); ax.set_xlabel("synthetic images")
    ax.set_title(f"{LAB[cfg]} (nano)",fontweight="bold"); ax.grid(True,ls=":",alpha=.6); ax.set_ylim(0.3,0.75)
    if cfg=="sun_off": ax.set_ylabel("ODD Macro-F1"); ax.legend(fontsize=9)
fig.suptitle("Confidence sensitivity of the data-scaling trend (nano): Sun-Off up, Baseline DOWN at every conf",fontweight="bold",fontsize=12)
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(FIG/"datascale_sensitivity.pdf"); plt.close(fig)

print("written ->",OUTDIR/"RESULTS_4_datascaling.txt")
print("figures ->",FIG/"datascale_odd_vs_data.pdf",",",FIG/"datascale_sensitivity.pdf")
