#!/usr/bin/env python3
"""Aggregate the multi-seed variance study.

Reads  eval_results/variance/seed{42..46}/{summary.csv, real_summary*.csv}
       eval_results/variance/within_rep/{...}                (sigma_within 2nd realization)
Writes eval_results/variance/variance_table.csv  (per-seed values + mean +- std, every config/conf/domain/metric)
Prints a headline summary (macro-F1 @ conf 0.25, ODD then IDD) and the sigma_within deltas.

Per-seed columns are kept explicit so each run can be tied back to the paper's reported numbers.
"""
import csv, glob, statistics
from pathlib import Path

REPO  = Path(__file__).resolve().parents[1]
VAR   = REPO / "eval_results" / "variance"
SEEDS = [42, 43, 44, 45, 46]
METRICS = ["macro_f1", "pfm1_recall", "starfish_recall"]


def read_dir(d: Path):
    """{(config, conf_float, domain): {metric: float}} from summary.csv + real_summary*.csv in d."""
    out = {}
    files = sorted(glob.glob(str(d / "summary.csv"))) + sorted(glob.glob(str(d / "real_summary*.csv")))
    for fn in files:
        with open(fn) as f:
            for row in csv.DictReader(f):
                try:
                    key = (row["config"], float(row["conf"]), row["domain"])
                except (KeyError, ValueError):
                    continue
                out[key] = {m: float(row[m]) for m in METRICS if row.get(m) not in (None, "")}
    return out


# ---- gather per-seed ----
per_seed = {}
for s in SEEDS:
    d = VAR / f"seed{s}"
    if (d / "summary.csv").exists() or glob.glob(str(d / "real_summary*.csv")):
        per_seed[s] = read_dir(d)
present = sorted(per_seed)
if not present:
    raise SystemExit(f"No per-seed eval found under {VAR}. Run eval_variance.sh first.")
print(f"[variance] seeds found: {present}")

keys = set()
for s in present:
    keys |= set(per_seed[s])

# ---- aggregate ----
rows = []
for (cfg, conf, dom) in sorted(keys):
    for m in METRICS:
        dval = {s: per_seed[s][(cfg, conf, dom)][m] for s in present
                if (cfg, conf, dom) in per_seed[s] and m in per_seed[s][(cfg, conf, dom)]}
        if not dval:
            continue
        v = list(dval.values())
        row = {"config": cfg, "conf": conf, "domain": dom, "metric": m,
               "mean": round(statistics.mean(v), 4),
               "std": round(statistics.stdev(v), 4) if len(v) > 1 else 0.0,
               "n_seeds": len(v)}
        for s in SEEDS:
            row[f"seed{s}"] = round(dval[s], 4) if s in dval else ""
        rows.append(row)

# ---- write csv ----
VAR.mkdir(parents=True, exist_ok=True)
out_csv = VAR / "variance_table.csv"
cols = ["config", "conf", "domain", "metric"] + [f"seed{s}" for s in SEEDS] + ["mean", "std", "n_seeds"]
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"[written] {out_csv}  ({len(rows)} rows)")


# ---- headline print ----
def headline(dom):
    sel = [r for r in rows if r["metric"] == "macro_f1" and abs(r["conf"] - 0.25) < 1e-9 and r["domain"] == dom]
    if not sel:
        return
    sel.sort(key=lambda r: -r["mean"])
    print(f"\n===== macro-F1 @ conf 0.25 -- {dom.upper()}  (mean +- std over seeds {present}) =====")
    hdr = f"{'config':<12} " + " ".join(f"{'s'+str(s):>7}" for s in SEEDS) + "     mean +- std"
    print(hdr); print("-" * len(hdr))
    for r in sel:
        cells = " ".join((f"{r[f'seed{s}']:>7.3f}" if r[f'seed{s}'] != "" else f"{'--':>7}") for s in SEEDS)
        print(f"{r['config']:<12} {cells}   {r['mean']:.3f} +- {r['std']:.3f}")


for dom in ("odd", "idd"):
    headline(dom)

# ---- sigma_within: canonical (seed 42) vs 2nd realization ----
wr = read_dir(VAR / "within_rep")
s42 = per_seed.get(42, {})
if wr and s42:
    print("\n===== sigma_within (seed 42: canonical vs 2nd realization) -- macro_f1 =====")
    print(f"{'config':<13} {'domain':<5} {'conf':<5} {'canon':>8} {'rep2':>8} {'|delta|':>8}")
    for (cfg, conf, dom), mv in sorted(wr.items()):
        a = s42.get((cfg, conf, dom), {}).get("macro_f1")
        if a is None or "macro_f1" not in mv:
            continue
        b = mv["macro_f1"]
        print(f"{cfg:<13} {dom:<5} {conf:<5} {a:8.4f} {b:8.4f} {abs(a - b):8.4f}")

print("\nDone. -> error-bar figure: feed variance_table.csv rows (mean,std) per config.")
