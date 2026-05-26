import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_factory.data_loader import get_loader_segment
from model.InfoFlow import InfoFlow, infoflow_loss


class EarlyStopping:
    def __init__(self, patience=3, dataset_name="", delta=0.0):
        self.patience = patience
        self.dataset = dataset_name
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            return
        if score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return
        self.best_score = score
        self.save_checkpoint(val_loss, model, path)
        self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        os.makedirs(path, exist_ok=True)
        ckpt = os.path.join(path, f"{self.dataset}_infoflow_checkpoint.pth")
        torch.save(model.state_dict(), ckpt)
        print(f"Validation loss decreased. Saving model to {ckpt}; val_loss={val_loss:.7f}")


def adjust_learning_rate(optimizer, epoch, lr, lr_decay="none"):
    if lr_decay == "none":
        return
    next_lr = lr * (0.5 ** max(epoch - 1, 0))
    for param_group in optimizer.param_groups:
        param_group["lr"] = next_lr
    print(f"Updating learning rate to {next_lr}")


def normalize_np(values, eps=1e-8):
    values = values.reshape(-1)
    return (values - values.min()) / (values.max() - values.min() + eps)


def combine_scores(reconstruction_error, nll, alpha):
    rec = normalize_np(reconstruction_error)
    density = normalize_np(nll)
    return alpha * density + (1.0 - alpha) * rec


def paper_threshold(scores):
    scores = scores.reshape(-1)
    return float(scores.mean() + scores.std())


def percentile_threshold(scores, anomaly_ratio):
    return float(np.percentile(scores.reshape(-1), 100.0 - anomaly_ratio))


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


class InfoFlowSolver(object):
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
        self.test_loader = get_loader_segment(
            self.data_path,
            batch_size=self.batch_size,
            win_size=self.win_size,
            step=self.step,
            mode="test",
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
        self.model = InfoFlow(
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

    def run_epoch(self, loader, train=False):
        self.model.train(train)
        losses = []
        for input_data, _ in loader:
            x = input_data.float().to(self.device)
            if train:
                self.optimizer.zero_grad()
            outputs = self.model(x)
            loss, parts = infoflow_loss(outputs, beta=self.beta, gamma=self.gamma)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            losses.append(float(loss.detach().cpu()))
        return float(np.average(losses)) if losses else 0.0

    def train(self):
        print("======================INFOFLOW TRAIN MODE======================")
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
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr, self.lr_decay)

    def collect_raw_scores(self, loader):
        self.model.eval()
        rec_errors = []
        nlls = []
        labels = []
        with torch.no_grad():
            for input_data, batch_labels in loader:
                x = input_data.float().to(self.device)
                outputs = self.model(x)
                rec_errors.append(outputs["reconstruction_error"].detach().cpu().numpy().reshape(-1))
                nlls.append(outputs["nll"].detach().cpu().numpy().reshape(-1))
                labels.append(batch_labels.detach().cpu().numpy().reshape(-1))
        rec_errors = np.concatenate(rec_errors, axis=0)
        nlls = np.concatenate(nlls, axis=0)
        labels = np.concatenate(labels, axis=0)
        return rec_errors, nlls, labels

    def test(self):
        checkpoint = os.path.join(self.model_save_path, f"{self.dataset}_infoflow_checkpoint.pth")
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        print("======================INFOFLOW TEST MODE======================")

        train_rec, train_nll, _ = self.collect_raw_scores(self.train_loader)
        test_rec, test_nll, test_labels = self.collect_raw_scores(self.thre_loader)

        if self.threshold_mode == "percentile":
            train_scores = combine_scores(train_rec, train_nll, self.alpha)
            test_scores = combine_scores(test_rec, test_nll, self.alpha)
            threshold_scores = np.concatenate([train_scores, test_scores], axis=0)
            threshold = percentile_threshold(threshold_scores, self.anormly_ratio)
        else:
            train_scores = combine_scores(train_rec, train_nll, self.alpha)
            test_scores = combine_scores(test_rec, test_nll, self.alpha)
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
        result_file = os.path.join(self.result_path, f"{self.dataset}_infoflow_metrics.json")
        with open(result_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {result_file}")
        return metrics
