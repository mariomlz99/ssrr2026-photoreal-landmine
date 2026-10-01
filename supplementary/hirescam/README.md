# Supplementary qualitative HiResCAM analysis

This folder contains the qualitative HiResCAM visualization that was present in an earlier manuscript draft and was removed from the final 8-page camera-ready paper for space and to avoid over-interpreting a very small number of selected examples.

## What is shown

The maps are HiResCAM visualizations computed on the stride-8 feature map feeding the YOLO11l detection head, with respect to the highest mine-class score.

- **Top row:** one PFM-1 example from the SULAND OOD split.
- **Bottom row:** one PMA-2 example from the SULAND OOD split.
- The two compared detectors are the **real-data-trained YOLO11l** and the **synthetic Baseline-30k YOLO11l**.
- Green dashed boxes denote SULAND ground truth.
- Blue boxes denote synthetic-model detections at confidence 0.50.
- In these selected examples the real-data model produces no detection at confidence 0.50.
- The values shown under the panels are the mean attention inside the ground-truth box relative to the image mean, together with the detection outcome.

The examples suggest that the real-data model can still concentrate attention on the target region even when it does not produce a confident detection. **This is qualitative, illustrative evidence only. It is not used as a calibration analysis and should not be interpreted as establishing a general confidence-versus-localization mechanism.**

## Files

- `fig5_hirescam_ood_examples.png` — convenient GitHub/README rendering.
- `fig5_hirescam_ood_examples.pdf` — vector/high-quality original figure.

## Method reference

R. L. Draelos and L. Carin, “Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks,” arXiv preprint arXiv:2011.08891, 2020.

> Note: although this figure is sometimes informally referred to as the “Grad-CAM figure,” the method used here is **HiResCAM**.
