# Path overrides for the train / eval scripts.
#   cp configs/env.example.sh configs/env.sh   &&   edit the paths below   &&   source configs/env.sh
#
# All optional: if a variable is unset, the scripts default to a REPO-RELATIVE path
# (i.e. <repo>/SULAND, <repo>/datasets, <repo>/runs, <repo>/eval_results), so the
# repo runs out-of-the-box if you place the data inside it. Override here to point elsewhere.

# Real dataset root: must contain  iid/ITA.yolo/{train,val,test}  and  ood/USA.yolo/val
export SSRR_DATA=/path/to/SULAND

# Synthetic datasets root: must contain  {config}_{10k,20k,30k}/{train,val}/{images,labels}
export SSRR_DATASETS=/path/to/synthetic_datasets

# Base dir holding runs/detect/{synthetic_ablation,real_ablation}/<run>/weights/best.pt
# (defaults to the repo root)
export SSRR_RUNS_BASE=/path/to/output_base

# (optional) explicit data.yaml for the real model training
# export SSRR_REAL_YAML=$SSRR_DATA/iid/ITA.yolo/ITA_train_val_2.yaml

# (optional) interpreter / cache mode / workers for reproduce_all.sh
# export SSRR_PY=/path/to/venv/bin/python3
# export SSRR_CACHE=ram          # disk (default, portable) | ram (fast, ~36 GB/run) | False
# export SSRR_WORKERS=4

# (optional) where eval CSVs / analysis outputs go (default: <repo>/eval_results and <repo>/results)
# export SSRR_EVAL=$PWD/eval_results
# export SSRR_OUT=$PWD/results
