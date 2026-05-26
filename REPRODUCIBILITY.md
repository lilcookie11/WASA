# Reproducibility Notes

This document records the benchmark configuration used by the released InfoFlow code.

## Datasets

The experiments use three public multivariate time-series anomaly detection datasets:

- PSM
- MSL
- SMD

Download the benchmark files with:

```bash
python3 tools/download_tslib_anomaly.py --datasets PSM MSL SMD
```

## Default Configuration

The paper uses the following core settings:

| Parameter | Value |
| --- | ---: |
| Window size | 100 |
| Learning rate | 1e-4 |
| Weight decay | 2e-2 |
| Maximum epochs | 10 |
| Beta | 0.6 |
| Gamma | 0.4 |
| Alpha | 0.5 |

The benchmark scripts in `scripts/InfoFlow_*.sh` contain the dataset-specific settings used for the released reproduction.

## Commands

Run all reported datasets:

```bash
bash scripts/InfoFlow_all.sh
```

Run a single dataset:

```bash
bash scripts/InfoFlow_PSM.sh
bash scripts/InfoFlow_MSL.sh
bash scripts/InfoFlow_SMD.sh
```

Run component tests:

```bash
python3 tests/test_infoflow_components.py
```

## Reported Paper Results

| Dataset | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| PSM | 97.13 | 98.90 | 98.00 |
| MSL | 92.35 | 96.03 | 94.15 |
| SMD | 89.65 | 96.15 | 92.64 |

All metrics are percentages and use point-adjusted anomaly detection evaluation.
