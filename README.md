# Smart Dirt Buildup Detection for Conveyor Belt Systems
### Binary Image Classification in Mining Environments

**Authors:** Erick Chauke, Dr Milka Madahana, & Dr John Ekoru  
**Dataset:** [IEEE DataPort](https://ieee-dataport.org/open-access/dirt-buildup-belt-conveyor-structures)

---

## Overview

Mining operations rely on long conveyor belt systems to move ore across distances.
Over time, dirt and material accumulates on the belt structures. When undetected,
rollers seize up, belts drift, and fires can start. A single unplanned stoppage
costs a mine tens of thousands of dollars per hour.

This project trains a model to classify images of conveyor belt structures as either
clean or carrying dirt buildup. The dataset contains 388 labelled photographs from
a real mining environment.

---

## Results

| Model                        | Accuracy | F1 Score |
|------------------------------|----------|----------|
| Scratch CNN (baseline)       | 85.71%   | 0.8000   |
| ResNet-50 (head only)        | 81.63%   | 0.7805   |
| EfficientNet-B0 (head only)  | 81.63%   | 0.7692   |
| ResNet-50 (fine-tuned)       | 85.71%   | 0.8108   |
| EfficientNet-B0 (fine-tuned) | **93.88%** | **0.9268** |
| Santos et al. (2020) benchmark | 89.75% | 0.8773   |

EfficientNet-B0 fully fine-tuned beat the published benchmark on both metrics.

---

## Approach

- Stratified train/validation/test split (70/18/12)
- Preprocessing with CLAHE brightness correction and ImageNet normalisation
- Augmentation on training set only (flip, rotate, brightness, blur)
- Two pretrained models trained in two stages: head only, then full fine-tuning
- Differential learning rates, label smoothing, and early stopping during fine-tuning
- 5-fold cross-validation on the best model
- Confidence and per-class error analysis

---

## Project Structure

```
conveyor-dirt-detection/
├── data/               <- images (not committed, download from IEEE DataPort)
├── outputs/            <- saved model checkpoints (not committed)
├── notebook.ipynb      <- full project notebook
├── notebook.html       <- rendered HTML version
└── README.md
```

---

## How to Run

1. Download the dataset from [IEEE DataPort](https://ieee-dataport.org/open-access/dirt-buildup-belt-conveyor-structures)
2. Place images in `data/clean/` and `data/dirty/`
3. Open `notebook.ipynb` and run all cells top to bottom

**Requirements:** Python 3.10+, PyTorch, timm, Albumentations, scikit-learn,
matplotlib, seaborn, Pillow, opencv-python, pandas

---

## Reference

L. H. Santos, A. Rocha, R. Reis, and F. G. Guimaraes, "Automatic System for Visual
Detection of Dirt Buildup on Conveyor Belts Using Convolutional Neural Networks,"
Sensors, vol. 20, no. 20, p. 5762, Oct. 2020.
https://doi.org/10.3390/s20205762
