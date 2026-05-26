import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_factory.data_loader import get_loader_segment
from model.AnomalyInfoFlow import AnomalyInfoFlow
from solver import my_kl_loss
from solver_infoflow import EarlyStopping, adjustment, combine_scores, paper_threshold, percentile_threshold


def normalize(values, eps=1e-8):
    values = values.reshape(-1)
    return (values - values.min()) / (values.max() - values.min() + eps)


class AnomalyInfoFlowSolver(object):
    def __init__(self, config):
        self.__dict__.update(config)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,
            mode="train",
            dataset=self.dataset,
        )
        self.vali_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,
            mode="val",
            dataset=self.dataset,
        )
        self.thre_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.win_size,
            mode="thre",
            dataset=self.dataset,
        )
        self.build_model()

    def build_model(self):
        self.model = AnomalyInfoFlow(
            win_size=self.win_size,
            enc_in=self.input_c,
            c_out=self.output_c,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            flow_layers=self.flow_layers,
            bottleneck_dim=self.bottleneck_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        self.criterion = nn.MSELoss()

    def association_losses(self, series, prior):
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            normalized_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                1, 1, 1, self.win_size
            )
            series_loss += torch.mean(my_kl_loss(series[u], normalized_prior.detach()))
            series_loss += torch.mean(my_kl_loss(normalized_prior.detach(), series[u]))
            prior_loss += torch.mean(my_kl_loss(normalized_prior, series[u].detach()))
            prior_loss += torch.mean(my_kl_loss(series[u].detach(), normalized_prior))
        return series_loss / len(prior), prior_loss / len(prior)

    def association_energy(self, outputs):
        series = outputs["series"]
        prior = outputs["prior"]
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            normalized_prior = prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(
                1, 1, 1, self.win_size
            )
            if u == 0:
                series_loss = my_kl_loss(series[u], normalized_prior.detach()) * self.temperature
                prior_loss = my_kl_loss(normalized_prior, series[u].detach()) * self.temperature
            else:
                series_loss += my_kl_loss(series[u], normalized_prior.detach()) * self.temperature
                prior_loss += my_kl_loss(normalized_prior, series[u].detach()) * self.temperature
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        return metric * outputs["reconstruction_error"]

    def run_epoch(self, loader, train=False):
        self.model.train(train)
        losses = []
        for input_data, _ in loader:
            x = input_data.float().to(self.device)
            if train:
                self.optimizer.zero_grad()
            outputs = self.model(x)
            info_regularizer = (
                self.beta * outputs["ib_reconstruction_loss"]
                + outputs["kl_loss"]
                + self.gamma * outputs["flow_nll"]
            )
            series_loss, prior_loss = self.association_losses(outputs["series"], outputs["prior"])
            rec_loss = self.criterion(outputs["reconstruction"], x)
            loss1 = rec_loss - self.k * series_loss + self.info_train_weight * info_regularizer
            loss2 = rec_loss + self.k * prior_loss + self.info_train_weight * info_regularizer
            if train:
                loss1.backward(retain_graph=True)
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            losses.append(float((loss1 + loss2).detach().cpu() / 2.0))
        return float(np.average(losses)) if losses else 0.0

    def train(self):
        print("======================ANOMALY-INFOFLOW TRAIN MODE======================")
        os.makedirs(self.model_save_path, exist_ok=True)
        early_stopping = EarlyStopping(patience=self.patience, dataset_name=self.dataset)
        train_steps = len(self.train_loader)
        for epoch in range(self.num_epochs):
            epoch_time = time.time()
            train_loss = self.run_epoch(self.train_loader, train=True)
            with torch.no_grad():
                val_loss = self.run_epoch(self.vali_loader, train=False)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} cost: {4:.2f}s".format(
                    epoch + 1,
                    train_steps,
                    train_loss,
                    val_loss,
                    time.time() - epoch_time,
                )
            )
            early_stopping(val_loss, self.model, self.model_save_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def collect_scores(self, loader):
        self.model.eval()
        association_scores = []
        info_scores = []
        labels = []
        with torch.no_grad():
            for input_data, batch_labels in loader:
                x = input_data.float().to(self.device)
                outputs = self.model(x)
                association_scores.append(self.association_energy(outputs).detach().cpu().numpy().reshape(-1))
                info_scores.append(
                    combine_scores(
                        outputs["reconstruction_error"].detach().cpu().numpy(),
                        outputs["nll"].detach().cpu().numpy(),
                        self.alpha,
                    )
                )
                labels.append(batch_labels.detach().cpu().numpy().reshape(-1))
        association_scores = np.concatenate(association_scores, axis=0)
        info_scores = np.concatenate(info_scores, axis=0)
        labels = np.concatenate(labels, axis=0)
        if self.association_weight >= 1.0:
            return association_scores, labels
        if self.association_weight <= 0.0:
            return info_scores, labels
        scores = self.association_weight * normalize(association_scores) + (1.0 - self.association_weight) * normalize(
            info_scores
        )
        return scores, labels

    def test(self):
        checkpoint = os.path.join(self.model_save_path, f"{self.dataset}_infoflow_checkpoint.pth")
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        print("======================ANOMALY-INFOFLOW TEST MODE======================")
        train_scores, _ = self.collect_scores(self.train_loader)
        test_scores, test_labels = self.collect_scores(self.thre_loader)
        if self.threshold_mode == "percentile":
            threshold = percentile_threshold(np.concatenate([train_scores, test_scores], axis=0), self.anormly_ratio)
        else:
            threshold = paper_threshold(train_scores)
        pred = (test_scores > threshold).astype(int)
        gt = test_labels.astype(int)
        gt, pred = adjustment(gt, pred)
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, _ = precision_recall_fscore_support(
            gt,
            pred,
            average="binary",
            zero_division=0,
        )
        metrics = {
            "dataset": self.dataset,
            "threshold_mode": self.threshold_mode,
            "threshold": float(threshold),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f_score),
        }
        print(
            "Accuracy : {accuracy:0.4f}, Precision : {precision:0.4f}, Recall : {recall:0.4f}, F-score : {f1:0.4f}".format(
                **metrics
            )
        )
        os.makedirs(self.result_path, exist_ok=True)
        result_file = os.path.join(self.result_path, f"{self.dataset}_anomaly_infoflow_metrics.json")
        with open(result_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {result_file}")
        return metrics
