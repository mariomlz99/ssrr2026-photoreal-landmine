# Appearance-adaptation control

The camera-ready paper treats the synthetic-only appearance-adaptation stage as part of the sim-to-real pipeline. To isolate its contribution, we retrained the Baseline-30k YOLO11l (seed 42) with this additional stage disabled while holding the standard YOLO augmentation pipeline and all other training settings fixed.

| Model | OOD mF1 @ 0.25 | AP50 | mAP50–95 |
|---|---:|---:|---:|
| Real-data YOLO11l | 0.348 | 0.322 | 0.181 |
| Synthetic, no appearance adaptation | 0.465 | 0.312 | 0.144 |
| Synthetic, full pipeline | 0.727 | 0.545 | 0.244 |

The control shows that appearance adaptation materially contributes to sim-to-real transfer. When it is removed, the synthetic model retains an advantage at the manuscript's fixed mF1 operating point, but not under standard AP50 or mAP50–95.

The no-adaptation model was trained for 100 epochs at 640×640 using the same Baseline-30k setup, seed 42, optimizer, standard YOLO augmentations, and training protocol as the released baseline. The extra synthetic-only Albumentations appearance-adaptation stage was the only intended change.

## Metrics and curves

- Compact results: [`metrics_summary.json`](metrics_summary.json) and [`summary.csv`](summary.csv)
- IID: [`standard metrics`](IID/standard_metrics.json), [`operating-point metrics`](IID/operating_point_metrics.json), and [`PR`](IID/BoxPR_curve.png) / [`F1`](IID/BoxF1_curve.png) / [`P`](IID/BoxP_curve.png) / [`R`](IID/BoxR_curve.png) curves
- OOD: [`standard metrics`](OOD/standard_metrics.json), [`operating-point metrics`](OOD/operating_point_metrics.json), and [`PR`](OOD/BoxPR_curve.png) / [`F1`](OOD/BoxF1_curve.png) / [`P`](OOD/BoxP_curve.png) / [`R`](OOD/BoxR_curve.png) curves
- Reproduction material: [`scripts/`](scripts/)

AP50 uses IoU 0.50 and mAP50–95 averages IoU thresholds from 0.50 to 0.95. To remain directly comparable with the manuscript's reported fixed-operating-point values, macro-F1 uses confidence 0.25 and the released evaluator's IoU matching threshold of 0.001.

The trained checkpoint is intentionally not stored in Git. Its SHA-256 is `35c15df686bdf614bead0b859a6c6846c32753662297850ef588a6ab6556c289`; an external download can be added alongside the other released checkpoint archives.
