import argparse
import os
import random

import numpy as np
import torch
from torch.backends import cudnn

from solver_anomaly_infoflow import AnomalyInfoFlowSolver
from utils.utils import mkdir


def main(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)
    solver = AnomalyInfoFlowSolver(vars(config))
    if config.mode == "train":
        solver.train()
    elif config.mode == "test":
        solver.test()
    elif config.mode == "train_test":
        solver.train()
        solver.test()
    return solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train_test", choices=["train", "test", "train_test"])
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataset", type=str, default="PSM", choices=["PSM", "MSL", "SMAP", "SMD"])
    parser.add_argument("--data_path", type=str, default="./dataset/PSM")
    parser.add_argument("--model_save_path", type=str, default="checkpoints_anomaly_infoflow")
    parser.add_argument("--result_path", type=str, default="results/anomaly_infoflow")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-2)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--input_c", type=int, default=25)
    parser.add_argument("--output_c", type=int, default=25)

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--flow_layers", type=int, default=4)

    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--beta", type=float, default=0.6)
    parser.add_argument("--gamma", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--association_weight", type=float, default=0.8)
    parser.add_argument("--info_train_weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=50.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--threshold_mode", type=str, default="percentile", choices=["paper", "percentile"])
    parser.add_argument("--anormly_ratio", type=float, default=1.0)

    config = parser.parse_args()
    args = vars(config)
    print("------------ Anomaly-InfoFlow Options -------------")
    for k, v in sorted(args.items()):
        print("%s: %s" % (str(k), str(v)))
    print("-------------- End ----------------")
    main(config)
