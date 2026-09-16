import numpy as np
import matplotlib.pyplot as plt
import os

def plot_attention_heatmap(attn_weights, savepath=None, title=None, head_avg=True):
    """
    attn_weights: either
      - (heads, H, Ls) for a single aggregated map, or
      - (N_samples, heads, H, Ls) for many samples (we will average across samples),
      - or (heads, H, Ls, ...) in rare shapes (we will try to reduce to 3 dims).
    head_avg: if True average across heads and plot (H, Ls). If False, plot each head.

    Returns matplotlib Figure.
    """
    arr = np.asarray(attn_weights)

    # Collapse any extra trailing dimensions if present: we only expect (heads, H, Ls)
    # or (N, heads, H, Ls). If arr.ndim > 4 and the last dims collapse to Ls, try to combine.
    if arr.ndim > 4:
        # collapse leading dims into sample axis and trailing dims into Ls if possible
        # General strategy: combine all leading axes except the last two (H,Ls) into one sample axis.
        # Find the last two dims which should be (H, Ls) or (heads, H, Ls) depending on leading dims.
        # Try to interpret the last two dims as (H, Ls). If arr.ndim >= 4, assume arr shape (..., H, Ls)
        H = arr.shape[-2]
        Ls = arr.shape[-1]
        # combine everything before H into a single sample/head axis, then try to split heads if possible.
        arr = arr.reshape(-1, H, Ls)

    # Now arr.ndim is either 3 or 4
    if arr.ndim == 3:
        # ambiguous: is this (heads, H, Ls) or (N_samples, H, Ls)
        # Heuristic: if first dim < 32 assume it's heads; if first dim is large assume samples
        if arr.shape[0] <= 32:
            # treat as (heads, H, Ls)
            arr_heads_h_ls = arr
            arr_samples_heads = arr_heads_h_ls[np.newaxis, ...]  # (1, heads, H, Ls)
        else:
            # treat as (N_samples, H, Ls) -> add a singleton heads axis
            arr_samples_heads = arr[:, np.newaxis, :, :]  # (N, 1, H, Ls)
    elif arr.ndim == 4:
        # expected (N_samples, heads, H, Ls) or (heads, H, Ls, something)
        if arr.shape[1] <= 32:
            # (N_samples, heads, H, Ls)
            arr_samples_heads = arr
        else:
            # Possibly (heads, H, Ls, extra) -> try to collapse last dim
            # reshape to (heads, H, Ls) by averaging last axis
            arr_heads_h_ls = arr.mean(axis=-1)
            arr_samples_heads = arr_heads_h_ls[np.newaxis, ...]
    else:
        raise ValueError(f"Unsupported attn_weights ndim={arr.ndim} with shape {arr.shape}")

    # Average across samples
    arr_mean_samples = arr_samples_heads.mean(axis=0)  # (heads, H, Ls)
    heads, H, Ls = arr_mean_samples.shape

    if head_avg:
        avg = arr_mean_samples.mean(axis=0)  # (H, Ls)
        fig, ax = plt.subplots(figsize=(8, 3 + 0.3 * H))
        im = ax.imshow(avg, aspect='auto', cmap='viridis')
        ax.set_xlabel("Sentiment time index (past → recent)")
        ax.set_ylabel("Horizon query index")
        ax.set_title(title or "Attention (avg heads & samples)")
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    else:
        cols = min(4, heads)
        rows = int(np.ceil(heads / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * (3 + 0.3 * H)))
        axes = np.array(axes).reshape(-1)
        for i in range(heads):
            ax = axes[i]
            im = ax.imshow(arr_mean_samples[i], aspect='auto', cmap='viridis')
            ax.set_title(f"Head {i}")
            ax.set_xlabel("Sentiment time index")
            ax.set_ylabel("Horizon")
        for ax in axes[heads:]:
            ax.axis('off')
        fig.suptitle(title or "Attention per head")
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath, bbox_inches='tight')
    return fig

def plot_calibration(preds, targets, quantiles, savepath=None):
    """
    preds: (N, H, Q)
    targets: (N, H)
    Plots empirical coverage vs nominal quantile for each horizon.
    """
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    N, H, Q = preds.shape
    plt.figure(figsize=(6, 4 + 0.4 * H))
    for h in range(H):
        emp = []
        for qidx, q in enumerate(quantiles):
            emp_cov = (targets[:, h] <= preds[:, h, qidx]).mean()
            emp.append(emp_cov)
        plt.plot(quantiles, emp, marker='o', label=f"H={h+1}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='ideal')
    plt.xlabel("Nominal quantile")
    plt.ylabel("Empirical coverage")
    plt.legend()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight')
    return plt.gcf()