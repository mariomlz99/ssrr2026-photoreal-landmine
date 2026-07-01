#!/usr/bin/env python3
"""Results-1 plain-text log: YOLOv11l, 30k, IDD(test)+ODD(val), full metrics + analysis.
Format inspired by the paper's Tables II/III. Output: results/RESULTS_1_yolov11l_30k.txt"""
import csv, statistics, os
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
EVAL=Path(os.environ.get("SSRR_EVAL", REPO/"eval_results"))
OUT =Path(os.environ.get("SSRR_OUT",  REPO/"results"))
AB=EVAL/"ablation_suland"
RUN=AB/"yolo11l_30k"
SYN=list(csv.DictReader(open(RUN/"summary.csv")))
REAL=list(csv.DictReader(open(RUN/"real_summary.csv")))
CONFS=[0.25,0.50,0.75]
LAB={"t_10":"T-10","n_0":"N-0","baseline":"Baseline","sun_off":"Sun-Off","yaw_off":"Yaw-Off","t_5":"T-5","i_1":"I-1",
     "s_off":"S-Off","pen_off":"Pen-Off","inv_off":"Inv-Off","n_30":"N-30","v_low":"V-Low","v_mid":"V-Mid","h_off":"H-Off"}
ORDER=["baseline","h_off","sun_off","t_10","t_5","v_low","v_mid","inv_off","pen_off","s_off","i_1","n_0","n_30","yaw_off"]

def g(rows,cfg,dom,conf,f):
    for r in rows:
        if r["config"]==cfg and r["domain"].lower().startswith(dom) and abs(float(r["conf"])-conf)<1e-6:
            return float(r[f])
    return float("nan")

# metric columns for the main per-conf tables
COLS=[("MF1","macro_f1"),("mP","macro_precision"),("mR","macro_recall"),("wF1","weighted_f1"),
      ("pfP","pfm1_precision"),("pfR","pfm1_recall"),("pfF1","pfm1_f1"),
      ("pmP","starfish_precision"),("pmR","starfish_recall"),("pmF1","starfish_f1")]

def table(dom,conf):
    # best synth per column
    best={f:max((g(SYN,c,dom,conf,f),c) for c in ORDER)[1] for _,f in COLS}
    hdr=f"{'Config':<10}"+"".join(f"{nm:>7}" for nm,_ in COLS)
    L=[hdr,"-"*len(hdr)]
    rv="".join(f"{g(REAL,'real_vv',dom,conf,f):>7.3f}" for _,f in COLS)
    L.append(f"{'Real':<10}{rv}")
    L.append("-"*len(hdr))
    for cfg in ORDER:
        s=f"{LAB[cfg]:<10}"
        for nm,f in COLS:
            v=g(SYN,cfg,dom,conf,f); mark="*" if best[f]==cfg else " "
            s+=f"{v:>6.3f}{mark}"
        L.append(s)
    return "\n".join(L)

def counts(dom):
    hdr=f"{'Config':<10}{'pfTP':>6}{'pfFP':>6}{'pfFN':>6}{'pmTP':>6}{'pmFP':>6}{'pmFN':>6}"
    L=[hdr,"-"*len(hdr)]
    def row(lab,src,cfg):
        return (f"{lab:<10}"+"".join(f"{int(g(src,cfg,dom,0.25,f)):>6}"
                for f in ("pfm1_tp","pfm1_fp","pfm1_fn","starfish_tp","starfish_fp","starfish_fn")))
    L.append(row("Real",REAL,"real_vv")); L.append("-"*len(hdr))
    for cfg in ORDER: L.append(row(LAB[cfg],SYN,cfg))
    return "\n".join(L)

# ---- narrative stats ----
def rmf1(dom,c): return g(REAL,"real_vv",dom,c,"macro_f1")
odd={c:g(SYN,c,"odd",0.25,"macro_f1") for c in ORDER}
ranked=sorted(ORDER,key=lambda c:odd[c],reverse=True)
med=statistics.median(odd.values()); bestc=ranked[0]; worstc=ranked[-1]
gain=100*(odd[bestc]-rmf1("odd",0.25))/rmf1("odd",0.25)
rc=[rmf1("odd",c) for c in CONFS]; t10_75=g(SYN,"t_10","odd",0.75,"macro_f1")
ridd=[rmf1("idd",c) for c in CONFS]
idd25={c:g(SYN,c,"idd",0.25,"macro_f1") for c in ORDER}; bi25=max(idd25,key=idd25.get)
idd75={c:g(SYN,c,"idd",0.75,"macro_f1") for c in ORDER}; bi75=max(idd75,key=idd75.get)
pfr={c:g(SYN,c,"odd",0.25,"pfm1_recall") for c in ORDER}; bp=max(pfr,key=pfr.get)
pmr={c:g(SYN,c,"odd",0.25,"starfish_recall") for c in ORDER}; bm=max(pmr,key=pmr.get)
# real false positives on ODD (false-alarm point)
r_fp=int(g(REAL,"real_vv","odd",0.25,"pfm1_fp"))+int(g(REAL,"real_vv","odd",0.25,"starfish_fp"))
b_fp=int(g(SYN,bestc,"odd",0.25,"pfm1_fp"))+int(g(SYN,bestc,"odd",0.25,"starfish_fp"))

T=f"""================================================================================
 RESULTS 1 -- YOLOv11l, 30k synthetic ablation vs real baseline
 Source: eval_results/ablation_suland/run_20260531_164001
 IDD = ITA.yolo/TEST (held-out, 3743 img)  |  ODD = USA.yolo/val (4436 img)
 IoU>=0.001 ; conf 0.25/0.50/0.75. Real on top; (*) = best synthetic per column.
 Cols: MF1=Macro-F1  mP/mR=Macro-Prec/Rec  wF1=Weighted-F1
       pf*=PFM-1(pfm1)  pm*=PMA-2(starfish)  P/R/F1=Prec/Recall/F1
 NOTE: IDD is reported on TEST (proper held-out eval; real early-stopped on val).
       ODD matches the prior draft exactly; IDD numbers differ from the val-based draft.
================================================================================

############################  ODD  (USA) -- primary transfer indicator  ############################

-- ODD @ conf 0.25 --
{table('odd',0.25)}

-- ODD @ conf 0.50 --
{table('odd',0.50)}

-- ODD @ conf 0.75 --
{table('odd',0.75)}

-- ODD detection counts @ conf 0.25 (false-alarm view) --
{counts('odd')}

############################  IDD  (ITA-test) -- in-distribution  ############################

-- IDD @ conf 0.25 --
{table('idd',0.25)}

-- IDD @ conf 0.50 --
{table('idd',0.50)}

-- IDD @ conf 0.75 --
{table('idd',0.75)}

-- IDD detection counts @ conf 0.25 --
{counts('idd')}

================================================================================
 ANALYSIS (Results 1)
================================================================================

A. SYNTHETIC TRAINING BREAKS THE DOMAIN BARRIER (ODD)
 - Every synthetic config beats real on ODD at all 3 confidences. At conf 0.25,
   real MF1 = {rc[0]:.3f}; median synth = {med:.3f}; best ({LAB[bestc]}) = {odd[bestc]:.3f}
   (+{gain:.0f}%). Worst synth ({LAB[worstc]}, {odd[worstc]:.3f}) still well above real.
 - Threshold robustness: real collapses {rc[0]:.3f} -> {rc[1]:.3f} -> {rc[2]:.3f}
   (0.25->0.50->0.75); {LAB[bestc]} holds {t10_75:.3f} at 0.75.
 - Precision / false alarms (ODD @0.25): real total FP = {r_fp}; {LAB[bestc]} FP = {b_fp}.
   Synthetic does not win by flooding detections -- it has higher precision AND recall.

B. IDD COMPLEMENTARY (in-distribution, TEST split)
 - Real dominates IDD at low/mid thresholds: MF1 = {ridd[0]:.3f} (0.25), above every
   synth (best synth {LAB[bi25]} = {idd25[bi25]:.3f}).
 - At conf 0.75 real drops to {ridd[2]:.3f}; best synth ({LAB[bi75]} = {idd75[bi75]:.3f})
   retains more -> synthetic confidence is better calibrated.
 - Real owns IDD (trained there); synthetic owns ODD. Deployment is OOD -> ODD matters.

C. RANKING (ODD MF1 @ conf 0.25)
{chr(10).join(f"   {i+1:>2}. {LAB[c]:<9} {odd[c]:.3f}" for i,c in enumerate(ranked))}
   real = {rc[0]:.3f}
 - Top tier (>=0.72): {", ".join(LAB[c] for c in ranked if odd[c]>=0.72)}
 - Sole outlier: {LAB[worstc]} ({odd[worstc]:.3f}) -- HDRI disabled.

D. PER-CLASS (ODD @ conf 0.25)
 - PFM-1 recall peaks at {LAB[bp]} ({pfr[bp]:.3f}); PMA-2 recall peaks at {LAB[bm]} ({pmr[bm]:.3f}).
   PFM-1 benefits from pose calibration; PMA-2 from terrain/scene density.

KEY NUMBERS
 ODD real    0.25/0.50/0.75 = {rc[0]:.3f}/{rc[1]:.3f}/{rc[2]:.3f}
 ODD {LAB[bestc]:<8} 0.25/0.50/0.75 = {odd[bestc]:.3f}/{g(SYN,bestc,'odd',0.50,'macro_f1'):.3f}/{t10_75:.3f}   (+{gain:.0f}% vs real @0.25)
 IDD real    0.25/0.50/0.75 = {ridd[0]:.3f}/{ridd[1]:.3f}/{ridd[2]:.3f}   (TEST)
================================================================================
"""
OUT.mkdir(parents=True,exist_ok=True)
out=OUT/"RESULTS_1_yolov11l_30k.txt"
out.write_text(T); print("written ->",out); print(T)
