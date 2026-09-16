import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from data import load_consolidated, prepare_pooled_dataset
from models import PriceOnlyLSTM, ConcatLSTM, DualEncoderModel
from train import train_nn, set_seed
from utils import quantile_loss, empirical_coverage
from eval import block_bootstrap_metric, diebold_mariano
from viz import plot_attention_heatmap, plot_calibration
from harrv import HARQuantileBaseline

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CONFIG = {
    'past_len': 60,
    'horizons': [1,2,3,5,10],
    'quantiles': [0.1, 0.5, 0.9],
    'batch_size': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'test_frac': 0.2
}

def collate_tensors(Xv, Xs, Y):
    import torch
    return TensorDataset(torch.tensor(Xv, dtype=torch.float32),
                         torch.tensor(Xs, dtype=torch.float32),
                         torch.tensor(Y, dtype=torch.float32))

def _process_attn_batch(attn, batch_size):
    """
    Turn a single-batch attention output into a list of per-sample arrays of shape (heads, H, Ls).
    attn: numpy array (various shapes)
    batch_size: int (actual size of this batch)
    Returns: list_length == batch_size of arrays (heads, H, Ls)
    """
    a = np.asarray(attn)
    per_sample = []

    # Common shapes to expect:
    # (batch_size, heads, H, Ls)
    # (batch_size, H, Ls)  -> no head dimension
    # (1, heads, H, Ls)     -> one set for whole batch
    # (heads, H, Ls)        -> no batch dim, just heads
    # (H, Ls)               -> neither batch nor head dims

    if a.ndim == 4:
        # If first dim equals batch_size -> direct mapping
        if a.shape[0] == batch_size:
            for i in range(batch_size):
                per_sample.append(a[i])
            return per_sample
        # If first dim == 1, replicate to batch_size
        if a.shape[0] == 1:
            for _ in range(batch_size):
                per_sample.append(a[0])
            return per_sample
        # Unexpected leading dim: try to flatten leading dims if product matches batch_size
        prod_lead = a.shape[0]
        if prod_lead == batch_size:
            for i in range(batch_size):
                per_sample.append(a[i])
            return per_sample
        # Fallback: treat as single map and replicate
        for _ in range(batch_size):
            per_sample.append(a.mean(axis=0))
        return per_sample

    elif a.ndim == 3:
        # shape possibilities: (batch_size, H, Ls) or (heads, H, Ls) or (1, H, Ls)
        if a.shape[0] == batch_size:
            # add head dimension =1
            for i in range(batch_size):
                per_sample.append(a[i][np.newaxis, :, :])
            return per_sample
        # if a.shape[0] == 1 -> replicate for batch
        if a.shape[0] == 1:
            for _ in range(batch_size):
                per_sample.append(a[0][np.newaxis, :, :])
            return per_sample
        # if a.shape[0] <= 32 and not equal batch_size: likely heads dimension
        if a.shape[0] <= 32:
            for _ in range(batch_size):
                per_sample.append(a)
            return per_sample
        # fallback: replicate averaged map
        avg = a.mean(axis=0)
        for _ in range(batch_size):
            per_sample.append(avg[np.newaxis, :, :])
        return per_sample

    elif a.ndim == 2:
        # (H, Ls) -> no head nor batch dims, replicate for every sample and add head=1
        for _ in range(batch_size):
            per_sample.append(a[np.newaxis, :, :])
        return per_sample

    else:
        # unexpected shape: return zero arrays to avoid crashing, but log shape
        print(f"Warning: unexpected attention array shape: {a.shape}. Returning zeros.")
        for _ in range(batch_size):
            per_sample.append(np.zeros((1, 1, 1), dtype=float))
        return per_sample

def main():
    set_seed(CONFIG['seed'])
    df = load_consolidated("data/consolidated.csv")
    Xv, Xs, Y, idx_map = prepare_pooled_dataset(df, past_len=CONFIG['past_len'], horizons=CONFIG['horizons'])
    N = len(Xv)
    print("Total pooled samples:", N)
    split = int(N * (1 - CONFIG['test_frac']))
    Xv_train, Xv_test = Xv[:split], Xv[split:]
    Xs_train, Xs_test = Xs[:split], Xs[split:]
    Y_train, Y_test = Y[:split], Y[split:]

    # Standardize features using train set
    vol_mu, vol_std = Xv_train.mean(), Xv_train.std() + 1e-8
    sent_mu, sent_std = Xs_train.mean(axis=(0,1)), Xs_train.std(axis=(0,1)) + 1e-8
    Xv_train_scaled = (Xv_train - vol_mu) / vol_std
    Xv_test_scaled = (Xv_test - vol_mu) / vol_std
    Xs_train_scaled = (Xs_train - sent_mu) / sent_std
    Xs_test_scaled = (Xs_test - sent_mu) / sent_std

    train_ds = collate_tensors(Xv_train_scaled, Xs_train_scaled, Y_train)
    test_ds = collate_tensors(Xv_test_scaled, Xs_test_scaled, Y_test)
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)

    quantiles = CONFIG['quantiles']
    H = len(CONFIG['horizons'])
    price_model = PriceOnlyLSTM(vol_input_dim=1, hidden_dim=32, d_model=64, num_horizons=H, quantiles=quantiles)
    concat_model = ConcatLSTM(input_dim=1+2, hidden_dim=32, d_model=64, num_horizons=H, quantiles=quantiles)
    dual_model = DualEncoderModel(vol_input_dim=1, sent_input_dim=2, vol_hidden=32, sent_hidden=32, d_model=64, num_horizons=H, quantiles=quantiles, n_heads=4)

    device = CONFIG['device']
    price_model = train_nn(price_model, train_loader, test_loader, quantiles, lr=1e-3, epochs=30, patience=6, device=device)
    concat_model = train_nn(concat_model, train_loader, test_loader, quantiles, lr=1e-3, epochs=30, patience=6, device=device)
    dual_model = train_nn(dual_model, train_loader, test_loader, quantiles, lr=1e-3, epochs=30, patience=6, device=device)

    # HAR baseline
    har = HARQuantileBaseline(windows=(1,5,22))
    har.fit(Xv_train.reshape(Xv_train.shape[0], Xv_train.shape[1], -1), Y_train, quantiles=quantiles)
    p_har, q_levels = har.predict(Xv_test.reshape(Xv_test.shape[0], Xv_test.shape[1], -1))

    # predictions with robust attention handling
    def predict_all(model, return_attn=False):
        model.eval()
        preds_list = []; targs_list = []; attn_per_sample = []
        with torch.no_grad():
            for vol, sent, targ in test_loader:
                vol = vol.to(device)
                sent = sent.to(device)
                batch_size = vol.shape[0]
                if isinstance(model, PriceOnlyLSTM):
                    out = model(vol).cpu().numpy()
                elif isinstance(model, ConcatLSTM):
                    inp = torch.cat([vol, sent], dim=-1)
                    out = model(inp).cpu().numpy()
                else:
                    if return_attn:
                        out, attn = model(vol, sent, return_attentions=True)
                        out = out.cpu().numpy()
                        # process attn for this batch into per-sample entries
                        try:
                            attn_np = attn.cpu().numpy()
                        except:
                            attn_np = np.asarray(attn)
                        per_sample_list = _process_attn_batch(attn_np, batch_size)
                        attn_per_sample.extend(per_sample_list)
                    else:
                        out = model(vol, sent).cpu().numpy()
                preds_list.append(out)
                targs_list.append(targ.numpy())
        preds = np.concatenate(preds_list, axis=0)
        targs = np.concatenate(targs_list, axis=0)
        if return_attn:
            # stack per-sample attention arrays into (N_samples, heads, H, Ls)
            attn_arr = np.stack(attn_per_sample, axis=0)
            return preds, targs, attn_arr
        return preds, targs

    p_price, y = predict_all(price_model)
    p_concat, _ = predict_all(concat_model)
    p_dual, y, attn_weights = predict_all(dual_model, return_attn=True)

    # compute pinball losses per quantile and horizon
    def compute_pinball_numpy(preds, targets, quantiles):
        Q = len(quantiles)
        H = preds.shape[1]
        losses = np.zeros((Q, H))
        for qidx, q in enumerate(quantiles):
            err = np.maximum((q - 1) * (targets - preds[:,:,qidx]), q * (targets - preds[:,:,qidx]))
            losses[qidx] = err.mean(axis=0)
        return losses

    losses_har = compute_pinball_numpy(p_har, Y_test, quantiles)
    losses_price = compute_pinball_numpy(p_price, y, quantiles)
    losses_concat = compute_pinball_numpy(p_concat, y, quantiles)
    losses_dual = compute_pinball_numpy(p_dual, y, quantiles)

    pooled = {
        'har': float(losses_har.mean()),
        'price': float(losses_price.mean()),
        'concat': float(losses_concat.mean()),
        'dual': float(losses_dual.mean())
    }
    print("Pooled mean pinball losses:", pooled)

    # ... (rest of evaluation, bootstraps, DM tests as before) ...

    # Attention visualization: attn_weights is (N_samples, heads, H, Ls)
    heatpath = os.path.join(RESULTS_DIR, "attention_avg_heatmap.png")
    plot_attention_heatmap(attn_weights, savepath=heatpath, title="Dual Encoder Attention (avg heads & samples)", head_avg=True)
    print("Saved attention heatmap to", heatpath)

    # Calibration plot
    calpath = os.path.join(RESULTS_DIR, "calibration_dual.png")
    plot_calibration(p_dual, y, quantiles, savepath=calpath)
    print("Saved calibration plot to", calpath)

if __name__ == "__main__":
    main()