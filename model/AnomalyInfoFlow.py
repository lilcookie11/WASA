import torch
import torch.nn as nn

from .AnomalyTransformer import AnomalyTransformer
from .InfoFlow import (
    ConditionalNormalizingFlow,
    VariationalCriticalInformationExtractor,
)
from .embed import DataEmbedding


class AnomalyInfoFlow(nn.Module):
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
        super(AnomalyInfoFlow, self).__init__()
        self.backbone = AnomalyTransformer(
            win_size=win_size,
            enc_in=enc_in,
            c_out=c_out,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
            output_attention=True,
        )
        self.condition_embedding = DataEmbedding(c_out, d_model, dropout)
        self.extractor = VariationalCriticalInformationExtractor(
            d_model=d_model,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )
        self.flow = ConditionalNormalizingFlow(
            data_dim=enc_in,
            cond_dim=d_model,
            hidden_dim=d_ff,
            flow_layers=flow_layers,
        )

    def forward(self, x):
        reconstruction, series, prior, sigmas = self.backbone(x)
        h = self.condition_embedding(reconstruction)
        critical_h, ib_terms = self.extractor(h)
        nll, flow_z, flow_log_det = self.flow(x, critical_h)
        reconstruction_error = torch.mean((x - reconstruction).pow(2), dim=-1)
        return {
            "reconstruction": reconstruction,
            "series": series,
            "prior": prior,
            "sigmas": sigmas,
            "reconstruction_error": reconstruction_error,
            "sequence_reconstruction_loss": reconstruction_error.mean(),
            "nll": nll,
            "flow_nll": nll.mean(),
            "flow_z": flow_z,
            "flow_log_det": flow_log_det,
            "critical_hidden": critical_h,
            **ib_terms,
        }
