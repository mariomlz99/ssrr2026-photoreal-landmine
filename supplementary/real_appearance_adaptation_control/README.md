# Matched real-data appearance-adaptation control

To distinguish synthetic-data effects from generic robustness provided by the additional appearance-adaptation stage, we retrained the real-data YOLO11l baseline with the exact same Albumentations stage used by the synthetic pipeline. The real IID training/validation split, YOLO11l initialization, seed 42, optimizer and schedule, batch size, image size, patience, dropout, and standard YOLO augmentations were unchanged.

Training stopped normally after epoch 39 under the unchanged patience-25 rule; the selected `best.pt` was from epoch 14.

## OOD comparison

| Training regime | mF1 @ 0.25 | AP50 | mAP50–95 |
|---|---:|---:|---:|
| Real baseline | 0.348 | 0.322 | 0.181 |
| Real + appearance adaptation | 0.601 | 0.477 | **0.275** |
| Synthetic Baseline-30k without adaptation | 0.465 | 0.312 | 0.144 |
| Synthetic Baseline-30k + adaptation | **0.727** | **0.545** | 0.244 |

## Fixed-confidence results

Real + appearance adaptation on OOD with the manuscript evaluator:

| Confidence | Macro-F1 |
|---:|---:|
| 0.25 | 0.601 |
| 0.50 | 0.529 |
| 0.75 | 0.031 |

Appearance adaptation provides substantial generic robustness in addition to supporting sim-to-real transfer. Under the same adaptation stage, the synthetic Baseline-30k remains higher than real-data training in OOD macro-F1 and AP50, whereas the adapted real model is higher in mAP50–95. The synthetic configurations also retain their fixed-confidence advantage across the evaluated thresholds.

This control does not establish that adaptation is exclusively synthetic-specific, that rendering diversity alone explains the full advantage, or that synthetic training dominates the matched real baseline under every metric.

## Metrics and reproduction material

- Four-way comparison: [`comparison.csv`](comparison.csv)
- Compact summary: [`metrics_summary.json`](metrics_summary.json)
- Confidence 0.50/0.75: [`fixed_confidence_results.txt`](fixed_confidence_results.txt) and [`confidence_050_075_metrics.json`](confidence_050_075_metrics.json)
- IID: [`standard metrics`](IID/standard_metrics.json), [`operating-point metrics`](IID/operating_point_metrics.json), and [`PR`](IID/BoxPR_curve.png) / [`F1`](IID/BoxF1_curve.png) / [`P`](IID/BoxP_curve.png) / [`R`](IID/BoxR_curve.png) curves
- OOD: [`standard metrics`](OOD/standard_metrics.json), [`operating-point metrics`](OOD/operating_point_metrics.json), and [`PR`](OOD/BoxPR_curve.png) / [`F1`](OOD/BoxF1_curve.png) / [`P`](OOD/BoxP_curve.png) / [`R`](OOD/BoxR_curve.png) curves
- Training summary: [`training_results.csv`](training_results.csv)
- Commands, configuration, and scripts: [`scripts/`](scripts/)

AP50 uses IoU 0.50 and mAP50–95 averages IoU thresholds from 0.50 to 0.95. The fixed-confidence macro-F1 values use the released manuscript evaluator's IoU matching threshold of 0.001. The OOD evaluator applies the required SULAND USA ground-truth class remapping.

The checkpoint is intentionally not stored in Git. Its SHA-256 is `9defbdd9af8f15c4488fc7ddf49af1591da9cb865f704e8ac2761b573631713b`; an external download can be added later.
