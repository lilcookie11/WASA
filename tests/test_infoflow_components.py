import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model.InfoFlow import InfoFlow, infoflow_loss, infoflow_score
from model.AnomalyInfoFlow import AnomalyInfoFlow


def test_infoflow_forward_returns_reconstruction_likelihood_and_ib_terms():
    torch.manual_seed(7)
    model = InfoFlow(
        win_size=16,
        enc_in=4,
        c_out=4,
        d_model=32,
        n_heads=4,
        e_layers=2,
        d_ff=64,
        flow_layers=2,
        bottleneck_dim=12,
        dropout=0.0,
    )
    x = torch.randn(3, 16, 4)

    out = model(x)

    assert out["reconstruction"].shape == x.shape
    assert out["nll"].shape == (3, 16)
    assert out["reconstruction_error"].shape == (3, 16)
    assert out["sequence_reconstruction_loss"].ndim == 0
    assert out["ib_reconstruction_loss"].ndim == 0
    assert out["kl_loss"].ndim == 0
    assert out["flow_nll"].ndim == 0


def test_infoflow_loss_backpropagates_through_all_components():
    torch.manual_seed(11)
    model = InfoFlow(
        win_size=8,
        enc_in=3,
        c_out=3,
        d_model=24,
        n_heads=3,
        e_layers=1,
        d_ff=48,
        flow_layers=1,
        bottleneck_dim=10,
        dropout=0.0,
    )
    x = torch.randn(2, 8, 3)

    out = model(x)
    loss, parts = infoflow_loss(out, beta=0.6, gamma=0.4)
    loss.backward()

    assert loss.ndim == 0
    assert set(parts) == {"sequence_reconstruction", "ib_reconstruction", "kl", "flow_nll", "total"}
    assert any(p.grad is not None for p in model.parameters())


def test_infoflow_score_combines_normalized_reconstruction_and_nll():
    reconstruction_error = torch.tensor([[1.0, 2.0, 3.0]])
    nll = torch.tensor([[3.0, 2.0, 1.0]])

    score = infoflow_score(reconstruction_error, nll, alpha=0.5)

    assert score.shape == (3,)
    assert torch.isfinite(score).all()
    assert torch.all(score >= 0)


def test_anomaly_infoflow_forward_keeps_association_and_adds_flow_terms():
    torch.manual_seed(17)
    model = AnomalyInfoFlow(
        win_size=8,
        enc_in=3,
        c_out=3,
        d_model=24,
        n_heads=3,
        e_layers=1,
        d_ff=48,
        flow_layers=1,
        bottleneck_dim=10,
        dropout=0.0,
    )
    x = torch.randn(2, 8, 3)

    out = model(x)

    assert out["reconstruction"].shape == x.shape
    assert len(out["series"]) == 1
    assert len(out["prior"]) == 1
    assert out["nll"].shape == (2, 8)


if __name__ == "__main__":
    test_infoflow_forward_returns_reconstruction_likelihood_and_ib_terms()
    test_infoflow_loss_backpropagates_through_all_components()
    test_infoflow_score_combines_normalized_reconstruction_and_nll()
    test_anomaly_infoflow_forward_keeps_association_and_adds_flow_terms()
