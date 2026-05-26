import argparse
import os

import numpy as np
import pandas as pd


def make_series(length, dims, seed, anomaly=False):
    rng = np.random.default_rng(seed)
    t = np.arange(length, dtype=np.float32)
    values = []
    for i in range(dims):
        values.append(np.sin(t / (7.0 + i)) + 0.25 * np.cos(t / (13.0 + i)))
    data = np.stack(values, axis=1) + rng.normal(0.0, 0.05, size=(length, dims))
    labels = np.zeros((length, 1), dtype=np.int64)
    if anomaly:
        start = length // 2
        end = min(length, start + max(8, length // 12))
        data[start:end] += 3.0
        labels[start:end] = 1
    return data.astype(np.float32), labels


def write_csv(path, name, values):
    columns = ["timestamp"] + [f"value_{i}" for i in range(values.shape[1])]
    frame = pd.DataFrame(np.concatenate([np.arange(len(values)).reshape(-1, 1), values], axis=1), columns=columns)
    frame.to_csv(os.path.join(path, name), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="dataset/synthetic_psm")
    parser.add_argument("--length", type=int, default=240)
    parser.add_argument("--dims", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    train, _ = make_series(args.length, args.dims, seed=1, anomaly=False)
    test, labels = make_series(args.length, args.dims, seed=2, anomaly=True)
    write_csv(args.output, "train.csv", train)
    write_csv(args.output, "test.csv", test)
    label_frame = pd.DataFrame(np.concatenate([np.arange(len(labels)).reshape(-1, 1), labels], axis=1), columns=["timestamp", "label"])
    label_frame.to_csv(os.path.join(args.output, "test_label.csv"), index=False)
    print(f"Wrote synthetic PSM-style data to {args.output}")


if __name__ == "__main__":
    main()
