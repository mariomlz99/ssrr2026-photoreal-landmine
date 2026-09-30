# Data setup

This repo ships **code only**. The two datasets are large and are **not** tracked in git
(`datasets/` and `SULAND/` are git-ignored). Download/obtain them and place them at the
repo root as described below; the scripts then resolve everything **with no configuration**,
because their defaults are repo-relative:

| Env var (optional override) | Default location | Holds |
|---|---|---|
| `SSRR_DATASETS` | `./datasets` | synthetic training sets |
| `SSRR_DATA`     | `./SULAND`   | real SULAND benchmark |
| `SSRR_RUNS_BASE`| `.` (repo root) | `runs/` + `eval_results/` outputs |

If your data lives elsewhere, copy `configs/env.example.sh` → `configs/env.sh`, edit the paths,
and `source configs/env.sh` before running.

---

## 1. Synthetic datasets  →  `./datasets/`

**Download** (≈439 GB, single archive `suland_synthetic_datasets.tar`): <https://ssrr-submission.s.gy/UFnfsN>

Extract at the repo root and verify integrity:

```bash
tar -xf suland_synthetic_datasets.tar -C /path/to/ssrr2026-photoreal-landmine   # archive root is datasets/
echo "b5ecd78f4f4c161d65bbd478016842420820450ab38800cd0d97621522c849f8  suland_synthetic_datasets.tar" | sha256sum -c
```

After extraction the tree is:

```
datasets/
├── baseline_30k/        v_low_30k/   v_mid_30k/   h_off_30k/   s_off_30k/
├── i_1_30k/             t_10_30k/    t_5_30k/     n_0_30k/     n_30_30k/
├── inv_off_30k/         sun_off_30k/ yaw_off_30k/ pen_off_30k/        # ← the 14 ablation configs @ 30k
├── baseline_10k/  baseline_20k/  sun_off_10k/  sun_off_20k/          # ← data-scaling (Table VI)
└── baseline_iddbal_30k/  baseline_volctrl_5304/                      # ← prevalence/volume controls (Sec IV-E)
```

Each dataset folder has this layout (labels are YOLO `.txt`; `data.yaml` is auto-generated on first run):

```
<dataset>/
├── train/ images/*.png   labels/*.txt
└── val/   images/*.png   labels/*.txt
```

- Classes: `0 = pfm1` (PFM-1), `1 = starfish` (PMA-2).
- 20 dataset folders total: 14 configs @ 30k + {baseline, sun_off} @ {10k, 20k} + 2 Sec IV-E controls.
- The 30k / 20k / 10k sets are strictly nested (10k ⊂ 20k ⊂ 30k).

---

## 2. Real benchmark (SULAND)  →  `./SULAND/`

SULAND is a published benchmark and is obtained from its **original source** (not redistributed here).
Place it so the tree is:

```
SULAND/
├── iid/ITA.yolo/
│   ├── train/  images/  labels/        # 22,756 imgs, real model trains here
│   ├── val/    images/  labels/        #  2,836 imgs, validation split
│   ├── test/   images/  labels/        #  3,743 imgs, IID evaluation (held-out)
│   └── ITA_train_val_2.yaml            # data.yaml for real-model training
└── ood/USA.yolo/
    └── val/    images/  labels/        #  4,436 imgs, OOD evaluation
```

- Images may be nested by video sequence (`.../images/<seq>/<frame>.jpg`); the loaders handle this.
- **IID = `ITA.yolo/test`** (the held-out split, not val).
- **OOD = `USA.yolo/val`** (the only USA split = the out-of-distribution test set).
- USA labels define `0=starfish, 1=butterfly` (opposite of synthetic); the eval applies a class remap automatically.

---

## 3. Trained weights (optional: reproduce the tables without training)

The released checkpoints let you reproduce every reported number by evaluation alone
(no training). Extract at the repo root; they unpack to
`runs/detect/{synthetic_ablation,real_ablation}/<run>/weights/best.pt`, exactly where
the eval scripts look. Each checkpoint also ships its `results.csv` (training curves).
Embedded training paths are anonymised; model weights are untouched.

| archive | models | reproduces | download | sha256 |
|---|---|---|---|---|
| `ssrr2026_weights_seed42.tar` (≈1.3 GB) | 43 (seed 42) | all paper tables | <https://ssrr-submission.s.gy/arZqug> | `fcb211233d72625a1b7955483966252f1138ac7211751623532cedaacb40c547` |
| `ssrr2026_weights_seeds43-46.tar` (≈1.4 GB) | 28 (seeds 43-46) | the seed-robustness study | <https://ssrr-submission.s.gy/3Hqcp3> | `009461956c71181f2e21fae9e53e63885d7d5a3c24faa4e3bd805ff9db4a5577` |

```bash
tar -xf ssrr2026_weights_seed42.tar -C /path/to/ssrr2026-photoreal-landmine   # -> runs/detect/...
# then evaluate (needs the SULAND real splits from Section 2):
bash eval_variance.sh        # or the per-table eval commands in the README
```

---

## 4. Sanity check

```bash
# synthetic: expect 20 dataset folders, each with train/ and val/
ls datasets | wc -l
ls datasets/baseline_30k/train/images | head

# real: expect the four splits
ls SULAND/iid/ITA.yolo            # train val test ITA_train_val_2.yaml
ls SULAND/ood/USA.yolo            # val
```

Once both are in place: `bash reproduce_all.sh` (full paper, seed 42) or `bash variance_study.sh within`
(multi-seed study). No further setup required.
