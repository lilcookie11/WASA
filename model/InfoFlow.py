import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embed import DataEmbedding


def _causal_mask(length, device):
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class CausalEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="gelu"):
        super(CausalEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        mask = _causal_mask(x.size(1), x.device)
        attn_out, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        return self.norm2(x + self.dropout(ff_out))


class TemporalFeatureEncoder(nn.Module):
    def __init__(self, enc_in, d_model, n_heads, e_layers, d_ff, dropout=0.1, activation="gelu"):
        super(TemporalFeatureEncoder, self).__init__()
        self.embedding = DataEmbedding(enc_in, d_model, dropout)
        self.layers = nn.ModuleList(
            [
                CausalEncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.embedding(x)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class VariationalCriticalInformationExtractor(nn.Module):
    def __init__(self, d_model, bottleneck_dim, dropout=0.1):
        super(VariationalCriticalInformationExtractor, self).__init__()
        self.to_mu = nn.Linear(d_model, bottleneck_dim)
        self.to_logvar = nn.Linear(d_model, bottleneck_dim)
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, h):
        mu = self.to_mu(h)
        logvar = self.to_logvar(h).clamp(min=-8.0, max=8.0)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        decoded_h = self.decoder(z)
        ib_reconstruction_loss = F.mse_loss(decoded_h, h)
        kl_loss = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
        return decoded_h, {
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "ib_reconstruction_loss": ib_reconstruction_loss,
            "kl_loss": kl_loss,
        }


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, data_dim, cond_dim, hidden_dim, mask):
        super(ConditionalAffineCoupling, self).__init__()
        self.register_buffer("mask", mask.view(1, 1, data_dim))
        self.net = nn.Sequential(
            nn.Linear(data_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, data_dim * 2),
        )

    def forward(self, x, cond):
        x_masked = x * self.mask
        scale_shift = self.net(torch.cat([x_masked, cond], dim=-1))
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = torch.tanh(scale) * (1.0 - self.mask)
        shift = shift * (1.0 - self.mask)
        y = x_masked + (1.0 - self.mask) * (x * torch.exp(scale) + shift)
        log_det = scale.sum(dim=-1)
        return y, log_det


class ConditionalNormalizingFlow(nn.Module):
    def __init__(self, data_dim, cond_dim, hidden_dim, flow_layers):
        super(ConditionalNormalizingFlow, self).__init__()
        layers = []
        for i in range(flow_layers):
            mask = torch.tensor(
                [(j + i) % 2 for j in range(data_dim)],
                dtype=torch.float32,
            )
            layers.append(ConditionalAffineCoupling(data_dim, cond_dim, hidden_dim, mask))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, cond):
        z = x
        log_det = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            z, layer_log_det = layer(z, cond)
            log_det = log_det + layer_log_det
        base_log_prob = -0.5 * (z.pow(2) + math.log(2.0 * math.pi)).sum(dim=-1)
        log_prob = base_log_prob + log_det
        nll = -log_prob
        return nll, z, log_det


class InfoFlow(nn.Module):
    def __init__(
        self,
        win_size,
        enc_in,
        c_out,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.0,
        activation="gelu",
        flow_layers=4,
        bottleneck_dim=128,
    ):
        super(InfoFlow, self).__init__()
        self.win_size = win_size
        self.enc_in = enc_in
        self.c_out = c_out
        self.encoder = TemporalFeatureEncoder(
            enc_in=enc_in,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        )
        self.extractor = VariationalCriticalInformationExtractor(
            d_model=d_model,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )
        self.projection = nn.Linear(d_model, c_out)
        self.flow = ConditionalNormalizingFlow(
            data_dim=enc_in,
            cond_dim=d_model,
            hidden_dim=d_ff,
            flow_layers=flow_layers,
        )

    def forward(self, x):
        h = self.encoder(x)
        critical_h, ib_terms = self.extractor(h)
        reconstruction = self.projection(critical_h)
        nll, flow_z, flow_log_det = self.flow(x, critical_h)
        reconstruction_error = torch.mean((x - reconstruction).pow(2), dim=-1)
        sequence_reconstruction_loss = F.mse_loss(reconstruction, x)
        flow_nll = nll.mean()
        return {
            "reconstruction": reconstruction,
            "reconstruction_error": reconstruction_error,
            "sequence_reconstruction_loss": sequence_reconstruction_loss,
            "nll": nll,
            "flow_nll": flow_nll,
            "flow_z": flow_z,
            "flow_log_det": flow_log_det,
            "hidden": h,
            "critical_hidden": critical_h,
            **ib_terms,
        }


def infoflow_loss(outputs, beta=0.6, gamma=0.4):
    sequence_reconstruction = outputs["sequence_reconstruction_loss"]
    ib_reconstruction = outputs["ib_reconstruction_loss"]
    kl = outputs["kl_loss"]
    flow_nll = outputs["flow_nll"]
    total = sequence_reconstruction + beta * ib_reconstruction + kl + gamma * flow_nll
    parts = {
        "sequence_reconstruction": float(sequence_reconstruction.detach().cpu()),
        "ib_reconstruction": float(ib_reconstruction.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "flow_nll": float(flow_nll.detach().cpu()),
        "total": float(total.detach().cpu()),
    }
    return total, parts


def _minmax_normalize(values, eps=1e-8):
    flat = values.reshape(-1)
    min_value = flat.min()
    max_value = flat.max()
    return (values - min_value) / (max_value - min_value + eps)


def infoflow_score(reconstruction_error, nll, alpha=0.5):
    rec = _minmax_normalize(reconstruction_error)
    density = _minmax_normalize(nll)
    score = alpha * density + (1.0 - alpha) * rec
    return score.reshape(-1)


def mean_std_threshold(scores):
    scores = scores.reshape(-1)
    return scores.mean() + scores.std(unbiased=False)
