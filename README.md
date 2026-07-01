# Towards Replacing Real Images with Synthetic Data for Surface Landmine Detection

Code to reproduce the results of the paper: training the **real** and **14 synthetic**
YOLOv11 detectors, evaluating them on the SULAND benchmark, and regenerating the result tables/figures.

- **IDD** (in-distribution) = ITA, evaluated on the held-out **test** split.
- **ODD** (out-of-distribution, zero-shot) = USA, evaluated on its **val** split.
- Detector: Ultralytics **YOLOv11** (l / s / n). Classes: `pfm1` (PFM-1), `starfish` (PMA-2).

> IDD numbers are reported on the held-out ITA/**test** split (never seen during training or
> validation). Synthetic models train on terrains gscatter_1-18 and validate on gscatter_19-20.

---

## Repository layout
```
training/    train_ablation.py        synthetic (14 configs x {n,s,l} x {10k,20k,30k}) + sim-to-real albumentations
             train_real_baseline.py   real YOLOv11l
             train_real_sizes.py       real YOLOv11 n/s
evaluation/  eval_ablation_suland.py        synthetic -> summary.csv (IDD test + ODD val, conf .25/.50/.75)
             eval_real_ablation_suland.py   real      -> real_summary[_s|_n].csv
             eval_iou_sweep_conf_0.25_0.5_0.75.py / eval_iou_sensitivity.py   IoU sweep (Results 2)
analysis/    make_results1..4_log.py   build the result .txt tables + figures from the eval CSVs
             make_variance_table.py    aggregate the 5-seed runs -> mean +- std
eval_results/  shipped eval CSVs (so the tables/figures regenerate without re-training)
results/       the produced RESULTS_*.txt + figs/   (reference outputs)
configs/       env.example.sh, ablation_configs.txt
reproduce_all.sh / variance_study.sh / eval_variance.sh   one-shot pipelines
DATA.md        how to obtain and place the datasets
```

## Results produced
| # | What | Builder | Output |
|---|------|---------|--------|
| 1 | YOLOv11l: real vs 14 synthetic, IDD/ODD, conf .25/.50/.75 | `analysis/make_results1_log.py` | `results/RESULTS_1_yolov11l_30k.txt` |
| 2 | IoU-sweep consistency (L models) | `analysis/make_results2_log.py` | `results/RESULTS_2_iou_sweep_yolov11l.txt` + figs |
| 3 | Detector-scale (deployability) n/s/l, conf .25/.50/.75 | `analysis/make_results3_log.py` | `results/RESULTS_3_deployability.txt` + figs |
| 4 | Data-scaling 10k/20k/30k (incl. nano inversion) | `analysis/make_results4_log.py` | `results/RESULTS_4_datascaling.txt` + figs |
| 5 | Seed robustness (5 seeds, headline configs) | `analysis/make_variance_table.py` | `eval_results/variance/variance_table.csv` |

---

## Setup
```bash
python -m venv venv && source venv/bin/activate
# install the torch build matching your CUDA first (see requirements.txt), then:
pip install -r requirements.txt
```

## Configure paths
Train/eval scripts read three environment variables (defaults point to the authors' machine).
Copy and edit:
```bash
cp configs/env.example.sh configs/env.sh   # edit the 3 paths
source configs/env.sh
```
- `SSRR_DATA`: real dataset root, `iid/ITA.yolo/{train,val,test}` and `ood/USA.yolo/val`
- `SSRR_DATASETS`: synthetic datasets root, `{config}_{10k,20k,30k}/{train,val}/{images,labels}`
- `SSRR_RUNS_BASE`: holds `runs/detect/{synthetic_ablation,real_ablation}/<run>/weights/best.pt`

---

## Reproduce

### A. Tables & figures only (no GPU, ~seconds)
The eval CSVs are shipped, so the result tables/figures regenerate directly:
```bash
python analysis/make_results1_log.py
python analysis/make_results2_log.py
python analysis/make_results3_log.py
python analysis/make_results4_log.py
# -> results/RESULTS_*.txt and results/figs/*.pdf
```

### B. Full pipeline (train -> eval -> analysis, GPU)
**1) Train.**
```bash
# real
python training/train_real_baseline.py            # YOLOv11l
python training/train_real_sizes.py --models n s   # YOLOv11 n, s
# synthetic (14 configs). L/30k for Results 1-2:
python training/train_ablation.py --models l --sizes 30k
# all sizes & data sizes for Results 3-4:
python training/train_ablation.py --models n s l --sizes 10k 20k 30k
```
**2) Evaluate** into the fixed layout the analysis reads (`eval_results/ablation_suland/yolo11<size>_<datasize>/`):
```bash
D=eval_results/ablation_suland
for sz in l s n; do for ds in 30k 20k 10k; do
  python evaluation/eval_ablation_suland.py --size $sz --datasize $ds --outdir $D/yolo11${sz}_${ds}
done; done
# real (no data-size axis) -> writes real_summary[_s|_n].csv into the 30k dirs
for sz in l s n; do
  python evaluation/eval_real_ablation_suland.py --size $sz --outdir $D/yolo11${sz}_30k
done
# IoU sweep for Results 2 (L, 30k)
python evaluation/eval_iou_sweep_conf_0.25_0.5_0.75.py --size l --datasize 30k \
       --outdir eval_results/iou_conf_sweep/yolo11l_30k
```
**3) Analysis.** Run the four `analysis/make_results*_log.py` as in step A.

Instead of running steps 1–3 by hand, you can run the whole train → eval → analysis pipeline
end-to-end with a single script: `bash reproduce_all.sh`. It executes all the commands above in
order and skips any step whose outputs already exist, so it is safe to re-run after an interruption.

### C. Seed robustness (5 seeds)
The headline configurations are re-run over five seeds to attach uncertainty:
```bash
bash variance_study.sh all     # train: seed 42 (full paper) then seeds 43-46 (headline configs)
bash eval_variance.sh          # eval + aggregate -> eval_results/variance/variance_table.csv
```
Any script honours `SSRR_RUN_SEED` (default 42) to set the run seed.

---

## Method notes (for fair comparison)
- **Identical training protocol** for real and synthetic: AdamW, lr0 3e-4, lrf 0.01, cos_lr,
  weight_decay 5e-4, 100 epochs, patience 25, batch 32, imgsz 640, seed 42, and the **same**
  built-in YOLO augmentations (hsv/translate/scale/fliplr/mosaic/erasing/randaugment).
- **Synthetic-only sim-to-real augmentation:** synthetic training additionally applies a
  photometric albumentations pipeline (sensor noise, blur, brightness/contrast/hue jitter, JPEG
  compression) to bridge the domain gap; it is part of the synthetic recipe, identical across
  all 14 configs, and is **not** applied to the real baseline (redundant for real imagery).
- **Seeded end-to-end** via `SSRR_RUN_SEED` (default 42): the run seed drives weight init,
  augmentation, and the sim-to-real recipe, so a fixed seed reproduces a run exactly. The
  headline tables use seed 42; the seed-robustness analysis repeats them over seeds 42-46.

## Data
The SULAND real splits and the rendered synthetic datasets / trained weights are not included
(size). Place them at the paths above (or set the env vars) and the pipeline runs end-to-end.
See **[`DATA.md`](DATA.md)** for the exact folder layout, expected structure, and a sanity check.

**Synthetic datasets** (the 20 sets used in the paper, ≈439 GB, `suland_synthetic_datasets.tar`):
<https://ssrr-submission.s.gy/UFnfsN>
`sha256: b5ecd78f4f4c161d65bbd478016842420820450ab38800cd0d97621522c849f8`
Extract at the repo root: `tar -xf suland_synthetic_datasets.tar -C .` → `./datasets/`.

**SULAND real benchmark**: obtained from its original source and placed at `./SULAND/` (see `DATA.md`).

**Trained weights** (optional): the 71 checkpoints are released so the tables reproduce by
evaluation alone, no training. Two archives (seed-42, ~1.3 GB; seeds 43-46, ~1.4 GB). Extract
into the repo (`runs/detect/.../weights/best.pt`) and evaluate. Links + checksums in `DATA.md`.

## Acknowledgements
This repository is anonymised for the initial submission.

## License
MIT (see `LICENSE`).
