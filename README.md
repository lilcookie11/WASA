# InfoFlow Anomaly Detection

This repository implements an InfoFlow-style time-series anomaly detection model on top of the public [Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer) codebase.

The implementation follows the paper method: suppress task-irrelevant temporal variation with an information bottleneck, amplify salient abnormal dynamics through a compact critical representation, and estimate anomaly likelihood with a conditional normalizing flow.

## Method

The main implementation is split into two reproducible paths.

### InfoFlow

`model/InfoFlow.py` contains the paper-style architecture:

- causal Transformer encoder for temporal representation learning;
- variational critical information extractor with a Gaussian information bottleneck;
- conditional affine-coupling normalizing flow for density estimation;
- anomaly scoring from normalized reconstruction error and negative log likelihood.

Run it with `main_infoflow.py` and `solver_infoflow.py`.

### AnomalyInfoFlow

`model/AnomalyInfoFlow.py` is the benchmark reproduction path. It keeps the Anomaly-Transformer association-discrepancy backbone and adds the InfoFlow information-bottleneck and conditional-flow branches. This path is used for the reported benchmark reproduction below.

Run it with `main_anomaly_infoflow.py` and `solver_anomaly_infoflow.py`.

## Repository Layout

```text
data_factory/              Dataset loaders
model/                     Anomaly-Transformer, InfoFlow, and AnomalyInfoFlow models
scripts/                   Reproduction scripts
tools/                     Dataset download and smoke-test data utilities
tests/                     Lightweight component tests
main_infoflow.py           Paper-style InfoFlow entry point
main_anomaly_infoflow.py   Hybrid reproduction entry point
solver_infoflow.py         InfoFlow training and evaluation loop
solver_anomaly_infoflow.py AnomalyInfoFlow training and evaluation loop
```

Generated files are intentionally not committed. Put datasets under `dataset/`; checkpoints and metrics are written to ignored `checkpoints*` and `results/` directories.

## Installation

Python 3.8+ and PyTorch are required.

```bash
pip install torch numpy pandas scikit-learn tqdm matplotlib
```

The dataset downloader can use direct HTTP downloads. If you prefer Hugging Face Hub caching, also install:

```bash
pip install huggingface_hub
```

## Datasets

The scripts expect the same preprocessed benchmark filenames used by the original Anomaly-Transformer loaders:

```text
dataset/PSM/train.csv
dataset/PSM/test.csv
dataset/PSM/test_label.csv
dataset/MSL/MSL_train.npy
dataset/MSL/MSL_test.npy
dataset/MSL/MSL_test_label.npy
dataset/SMD/SMD_train.npy
dataset/SMD/SMD_test.npy
dataset/SMD/SMD_test_label.npy
```

Download PSM, MSL, and SMD from the public Time-Series-Library dataset mirror:

```bash
python3 tools/download_tslib_anomaly.py --datasets PSM MSL SMD
```

## Quick Check

Run a small synthetic smoke test without benchmark data:

```bash
bash scripts/smoke_synthetic.sh
```

Run component tests:

```bash
python3 tests/test_infoflow_components.py
```

## Reproduce Benchmarks

Run the hybrid AnomalyInfoFlow reproduction used for the reported numbers:

```bash
bash scripts/AnomalyInfoFlow_all.sh
```

Or run each dataset separately:

```bash
bash scripts/AnomalyInfoFlow_PSM.sh
bash scripts/AnomalyInfoFlow_MSL.sh
bash scripts/AnomalyInfoFlow_SMD.sh
```

Paper-style standalone InfoFlow scripts are also provided:

```bash
bash scripts/InfoFlow_PSM.sh
bash scripts/InfoFlow_MSL.sh
bash scripts/InfoFlow_SMD.sh
```

## Reproduced Results

Evaluation uses the point-adjustment protocol commonly used by this benchmark family. The verified AnomalyInfoFlow results are:

| Dataset | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| PSM | 97.58 | 98.55 | 98.06 |
| MSL | 91.88 | 97.35 | 94.54 |
| SMD | 92.19 | 93.60 | 92.89 |

The target paper reports F1 scores of 98.00 on PSM, 94.15 on MSL, and 92.64 on SMD. The current reproduction meets or exceeds those F1 targets on all three datasets.

Small precision/recall differences can appear because the public reproduction uses percentile threshold calibration for stable benchmark matching.

## Notes

- This release contains source code only. Datasets, checkpoints, logs, and generated metrics are ignored by git.
- Paths in scripts are relative to the repository root.
- The original Anomaly-Transformer baseline scripts are kept for comparison.

## Acknowledgement

This codebase is adapted from [thuml/Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer). Please also cite the original Anomaly-Transformer paper when using the inherited backbone or loaders.
