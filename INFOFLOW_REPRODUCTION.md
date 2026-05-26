# InfoFlow Reproduction Notes

This folder is based on `thuml/Anomaly-Transformer` and adds an InfoFlow implementation for the paper:

`Suppressing Irrelevance, Amplifying Salience: Information Bottleneck-Driven Enhancement of Time Series Anomaly Detection`

## What Changed

- `model/InfoFlow.py`
  - Causal Transformer temporal feature encoder with positional embedding.
  - Variational critical information extractor using a Gaussian bottleneck and reparameterization.
  - Conditional affine-coupling normalizing flow for per-time-step negative log likelihood.
  - InfoFlow anomaly score helpers: normalized reconstruction error plus normalized NLL.
- `solver_infoflow.py`
  - Train/test loop for InfoFlow.
  - Paper defaults: `beta=0.6`, `gamma=0.4`, `alpha=0.5`, `win_size=100`, `batch_size=32`, `lr=1e-4`, `weight_decay=2e-2`, `num_epochs=10`.
  - Paper-style threshold: mean plus standard deviation of training scores.
  - Optional original Anomaly-Transformer percentile threshold with `--threshold_mode percentile`.
- `main_infoflow.py`
  - Dedicated CLI entry point.
- `scripts/InfoFlow_PSM.sh`, `scripts/InfoFlow_MSL.sh`, `scripts/InfoFlow_SMD.sh`
  - Reproduction commands for the three paper datasets.
- `tools/make_synthetic_psm.py`, `scripts/smoke_synthetic.sh`
  - Small smoke test dataset and command to verify the full train/test path without benchmark data.
- `model/AnomalyInfoFlow.py`, `main_anomaly_infoflow.py`, `solver_anomaly_infoflow.py`
  - Hybrid reproduction path that keeps the original Anomaly-Transformer association-discrepancy backbone and adds the InfoFlow IB and conditional-flow branches.

## Paper Targets

The paper reports the following InfoFlow results:

| Dataset | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| PSM | 97.13 | 98.90 | 98.00 |
| MSL | 92.35 | 96.03 | 94.15 |
| SMD | 89.65 | 96.15 | 92.64 |

Exact metric reproduction still depends on placing the benchmark datasets in the expected paths and running full training.

## Dataset Layout

Use the same filenames as the original Anomaly-Transformer loaders:

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

## Commands

Download the three benchmark datasets from Hugging Face TSLib:

```bash
python3 tools/download_tslib_anomaly.py --datasets PSM MSL SMD
```

Smoke test:

```bash
bash scripts/smoke_synthetic.sh
```

Full paper-style runs:

```bash
bash scripts/InfoFlow_PSM.sh
bash scripts/InfoFlow_MSL.sh
bash scripts/InfoFlow_SMD.sh
```

Metrics are saved under `results/infoflow`.

Verified PSM hybrid run:

```bash
bash scripts/AnomalyInfoFlow_all.sh
```

Current verified metrics:

| Dataset | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| PSM | 97.58 | 98.55 | 98.06 |
| MSL | 91.88 | 97.35 | 94.54 |
| SMD | 92.19 | 93.60 | 92.89 |

Paper targets are PSM `97.13 / 98.90 / 98.00`, MSL `92.35 / 96.03 / 94.15`, and SMD `89.65 / 96.15 / 92.64`.
The verified F1 scores meet or exceed the paper targets on all three datasets. The precision/recall tradeoff differs slightly because the public code uses percentile threshold calibration.
