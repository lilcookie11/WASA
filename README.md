# InfoFlow

Official implementation for the paper:

**Suppressing Irrelevance, Amplifying Salience: Information Bottleneck-Driven Enhancement of Time Series Anomaly Detection**

InfoFlow is an unsupervised time-series anomaly detection framework that improves temporal anomaly scoring by learning a compact critical representation. The model combines temporal association modeling, an information bottleneck module, and conditional flow-based likelihood estimation to suppress irrelevant variation while preserving anomaly-sensitive dynamics.

## Method Overview

InfoFlow contains three main components:

1. **Temporal dependency encoder.** A Transformer-based encoder captures long-range temporal dependencies and produces reconstruction-aware sequence representations.
2. **Critical information bottleneck.** A variational bottleneck extracts compact salient features and regularizes irrelevant information through reconstruction and KL objectives.
3. **Conditional normalizing flow.** A conditional affine-coupling flow estimates the likelihood of each time step conditioned on the learned critical representation.

The anomaly score combines reconstruction discrepancy, temporal association discrepancy, and flow-based negative log likelihood. The released benchmark scripts use the same point-adjustment protocol as prior time-series anomaly detection work.

## Repository Structure

```text
data_factory/                  Dataset loaders
model/InfoFlow.py              InfoFlow model used in the paper experiments
model/infoflow_modules.py      Information bottleneck and conditional-flow modules
model/AnomalyTransformer.py    Temporal association backbone
solver_infoflow.py             Training, scoring, and evaluation for InfoFlow
main.py                        Main entry point for InfoFlow
scripts/InfoFlow_*.sh          Reproduction scripts for PSM, MSL, and SMD
tools/download_tslib_anomaly.py Dataset download utility
tests/test_infoflow_components.py Lightweight model component tests
```

The original Anomaly-Transformer entry point is kept as `main_anomaly_transformer.py` for baseline comparison only.

## Requirements

The code was tested with Python 3 and PyTorch. Install the required packages with:

```bash
pip install -r requirements.txt
```

Optional dependency for Hugging Face Hub caching:

```bash
pip install huggingface_hub
```

## Data Preparation

The loaders expect the following preprocessed benchmark layout:

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

The public Time-Series-Library mirror can be downloaded with:

```bash
python3 tools/download_tslib_anomaly.py --datasets PSM MSL SMD
```

Datasets, checkpoints, logs, and generated metrics are intentionally excluded from git.

## Training and Evaluation

Run all benchmark experiments:

```bash
bash scripts/InfoFlow_all.sh
```

Run each dataset separately:

```bash
bash scripts/InfoFlow_PSM.sh
bash scripts/InfoFlow_MSL.sh
bash scripts/InfoFlow_SMD.sh
```

Metrics are saved under `results/infoflow_*`.

## Quick Verification

Run lightweight component tests:

```bash
python3 tests/test_infoflow_components.py
```

Run a smoke test on synthetic data:

```bash
bash scripts/smoke_synthetic.sh
```

## Main Results

The following table reports the InfoFlow results in the paper. Precision, recall, and F1 are reported in percent.

| Dataset | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| PSM | 97.13 | 98.90 | 98.00 |
| MSL | 92.35 | 96.03 | 94.15 |
| SMD | 89.65 | 96.15 | 92.64 |

The released scripts are configured to reproduce the paper-level performance on the same preprocessed benchmark splits. Minor deviations can occur across hardware, PyTorch versions, and stochastic training runs.

## Citation

If you use this repository, please cite the corresponding paper:

```bibtex
@inproceedings{infoflow,
  title     = {Suppressing Irrelevance, Amplifying Salience: Information Bottleneck-Driven Enhancement of Time Series Anomaly Detection},
  author    = {Mo, Yuhua and Ma, Yuhao and Liu, Xipeng and Wang, Jibin and Deng, Chao and Huang, Liying and Yang, Gang and Zhou, Fan},
  booktitle = {WASA},
  year      = {2026}
}
```

Please replace the venue metadata with the final camera-ready information when available.

## Acknowledgement

This implementation builds on the public Anomaly-Transformer codebase for the temporal association backbone and benchmark loaders.
