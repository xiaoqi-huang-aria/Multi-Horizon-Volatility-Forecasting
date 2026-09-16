import torch
import numpy as np

def quantile_loss(preds, targets, quantiles):
    """
    preds: (B, H, Q)
    targets: (B, H)
    quantiles: list-like of Q values
    returns mean pinball loss (scalar) and per-sample-per-horizon loss (B, H)
    """
    if not isinstance(quantiles, torch.Tensor):
        q = torch.tensor(quantiles, device=preds.device, dtype=preds.dtype)
    else:
        q = quantiles
    targets_exp = targets.unsqueeze(-1).expand_as(preds)
    errors = targets_exp - preds
    loss = torch.max((q - 1) * errors, q * errors)  # (B, H, Q)
    mean_loss = loss.mean()
    # per-sample per-horizon aggregated over quantiles (e.g., mean across Q)
    per_sample_horizon = loss.mean(dim=-1)  # (B, H)
    return mean_loss, per_sample_horizon

def empirical_coverage(preds, targets, quantiles):
    """
    preds: (N, H, Q)
    targets: (N, H)
    quantiles: list of Q levels
    returns coverage matrix (Q, H)
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    Q = preds.shape[2]
    H = preds.shape[1]
    cov = np.zeros((Q, H), dtype=float)
    for qidx in range(Q):
        cov[qidx] = (targets <= preds[:, :, qidx]).mean(axis=0)
    return cov