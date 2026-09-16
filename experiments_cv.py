#!/usr/bin/env python3
"""
Run as before:
  python experiments_cv.py --past-lens 20 60 120         # quick then full on top_k
  python experiments_cv.py --past-lens 20 60 120 --full  # same, but forces full stage
  python experiments_cv.py --help               
This file runs quick screening and full-stage experiments, aggregates results,
and saves per-fold and aggregated outputs under results_cv/.
"""
import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import load_consolidated, prepare_pooled_dataset
from models import PriceOnlyLSTM, ConcatLSTM, DualEncoderModel
from train import train_nn, set_seed
from harrv import HARQuantileBaseline
from utils import empirical_coverage
from eval import block_bootstrap_metric, diebold_mariano
from viz import plot_calibration, plot_attention_heatmap
from event_viz import select_event_dates_by_vol_spike, plot_event_attention_series
from model_utils import count_parameters, match_param_budget, save_model_size_report

ROOT_RESULTS = "results_cv"
os.makedirs(ROOT_RESULTS, exist_ok=True)

BASE_CONFIG = {
    'horizons': [1,2,3,5,10],
    'quantiles': [0.1, 0.5, 0.9],
    'batch_size': 128,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'n_folds': 5,
    'epochs_quick': 6,
    'epochs_full': 30,
    'patience': 6,
    'bootstrap_n': 2000,
    'block_size': 60,
    'top_k': 3
}

DEFAULT_PAST_LENS = [20, 60, 120]

# ---------------------- helpers ----------------------
def collate_tensors(Xv, Xs, Y):
    return TensorDataset(torch.tensor(Xv, dtype=torch.float32),
                         torch.tensor(Xs, dtype=torch.float32),
                         torch.tensor(Y, dtype=torch.float32))

def compute_pinball_matrix(preds, targets, quantiles):
    N, H, Q = preds.shape
    per_sample = np.zeros((N, H, Q), dtype=float)
    for qidx, q in enumerate(quantiles):
        e = targets - preds[:, :, qidx]
        per_sample[:, :, qidx] = np.maximum((q - 1) * e, q * e)
    loss_mat = per_sample.mean(axis=0).T  # (Q, H)
    return loss_mat, per_sample

def bootstrap_p_value(diff_series, block_size=60, n_boot=2000, seed=0):
    np.random.seed(seed)
    T = len(diff_series)
    if T <= 0:
        return 1.0, np.array([])
    n_blocks = int(np.ceil(T / block_size))
    boots = []
    for _ in range(n_boot):
        starts = np.random.randint(0, max(1, T - block_size + 1), size=n_blocks)
        samp = []
        for s in starts:
            samp.append(diff_series[s:s+block_size])
        samp = np.concatenate(samp)[:T]
        boots.append(samp.mean())
    boots = np.array(boots)
    p_one_sided = np.mean(boots >= 0)
    return p_one_sided, boots

def predict_all(model, data_loader, device, return_attn=False):
    model.eval()
    preds_list = []
    attn_list = []
    with torch.no_grad():
        for vol, sent, targ in data_loader:
            vol = vol.to(device); sent = sent.to(device)
            if isinstance(model, PriceOnlyLSTM):
                out = model(vol).cpu().numpy()
                preds_list.append(out)
            elif isinstance(model, ConcatLSTM):
                inp = torch.cat([vol, sent], dim=-1)
                out = model(inp).cpu().numpy()
                preds_list.append(out)
            else:
                if return_attn:
                    out, attn = model(vol, sent, return_attentions=True)
                    preds_list.append(out.cpu().numpy())
                    try:
                        a = attn.cpu().numpy()
                    except:
                        a = np.asarray(attn)
                    if a.ndim == 4 and a.shape[0] == out.shape[0]:
                        for i in range(a.shape[0]):
                            attn_list.append(a[i])
                    elif a.ndim == 3:
                        for _ in range(out.shape[0]):
                            attn_list.append(a)
                    elif a.ndim == 4 and a.shape[0] == 1:
                        for _ in range(out.shape[0]):
                            attn_list.append(a[0])
                    else:
                        avg = a.mean(axis=0)
                        for _ in range(out.shape[0]):
                            attn_list.append(avg)
                else:
                    out = model(vol, sent)
                    preds_list.append(out.cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)
    attn_arr = np.stack(attn_list, axis=0) if len(attn_list) > 0 else None
    return preds, attn_arr

# ---------------------- single fold train/eval ----------------------
def run_fold(Xv_train, Xs_train, Y_train, Xv_test, Xs_test, Y_test, cfg, model_params, quick=False, save_attn=False, out_dir=None, no_sentiment=False):
    device = cfg['device']
    quantiles = cfg['quantiles']
    H = len(cfg['horizons'])
    if no_sentiment:
        Xs_train = np.zeros_like(Xs_train)
        Xs_test = np.zeros_like(Xs_test)

    vol_mu, vol_std = Xv_train.mean(), Xv_train.std() + 1e-8
    sent_mu, sent_std = Xs_train.mean(axis=(0,1)), Xs_train.std(axis=(0,1)) + 1e-8
    Xv_tr_s = (Xv_train - vol_mu) / vol_std
    Xv_te_s = (Xv_test - vol_mu) / vol_std
    Xs_tr_s = (Xs_train - sent_mu) / sent_std
    Xs_te_s = (Xs_test - sent_mu) / sent_std

    train_ds = collate_tensors(Xv_tr_s, Xs_tr_s, Y_train)
    test_ds = collate_tensors(Xv_te_s, Xs_te_s, Y_test)
    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)

    price_params = model_params.get('price', {'hidden_dim':32, 'd_model':64})
    concat_params = model_params.get('concat', {'hidden_dim':32, 'd_model':64})
    dual_params = model_params.get('dual', {'vol_hidden':32, 'sent_hidden':32, 'd_model':64, 'n_heads':4})

    epochs = cfg['epochs_quick'] if quick else cfg['epochs_full']

    price_model = PriceOnlyLSTM(vol_input_dim=1, hidden_dim=price_params['hidden_dim'], d_model=price_params['d_model'], num_horizons=H, quantiles=quantiles)
    concat_model = ConcatLSTM(input_dim=1+2, hidden_dim=concat_params['hidden_dim'], d_model=concat_params['d_model'], num_horizons=H, quantiles=quantiles)
    dual_model = DualEncoderModel(vol_input_dim=1, sent_input_dim=2, vol_hidden=dual_params['vol_hidden'], sent_hidden=dual_params['sent_hidden'], d_model=dual_params['d_model'], num_horizons=H, quantiles=quantiles, n_heads=dual_params.get('n_heads',4))

    price_model = train_nn(price_model, train_loader, test_loader, quantiles, lr=1e-3, epochs=epochs, patience=cfg['patience'], device=device)
    concat_model = train_nn(concat_model, train_loader, test_loader, quantiles, lr=1e-3, epochs=epochs, patience=cfg['patience'], device=device)
    dual_model = train_nn(dual_model, train_loader, test_loader, quantiles, lr=1e-3, epochs=epochs, patience=cfg['patience'], device=device)

    har = HARQuantileBaseline(windows=(1,5,22))
    har.fit(Xv_train.reshape(Xv_train.shape[0], Xv_train.shape[1], -1), Y_train, quantiles=quantiles)
    p_har, _ = har.predict(Xv_test.reshape(Xv_test.shape[0], Xv_test.shape[1], -1))

    p_price, _ = predict_all(price_model, test_loader, device, return_attn=False)
    p_concat, _ = predict_all(concat_model, test_loader, device, return_attn=False)
    p_dual, attn_arr = predict_all(dual_model, test_loader, device, return_attn=True)

    losses_har, per_har = compute_pinball_matrix(p_har, Y_test, cfg['quantiles'])
    losses_price, per_price = compute_pinball_matrix(p_price, Y_test, cfg['quantiles'])
    losses_concat, per_concat = compute_pinball_matrix(p_concat, Y_test, cfg['quantiles'])
    losses_dual, per_dual = compute_pinball_matrix(p_dual, Y_test, cfg['quantiles'])

    cov_dual = empirical_coverage(p_dual, Y_test, cfg['quantiles'])
    cov_price = empirical_coverage(p_price, Y_test, cfg['quantiles'])
    cov_concat = empirical_coverage(p_concat, Y_test, cfg['quantiles'])
    cov_har = empirical_coverage(p_har, Y_test, cfg['quantiles'])

    if save_attn and (attn_arr is not None) and (out_dir is not None):
        np.savez_compressed(os.path.join(out_dir, "attn_fold.npz"), attn=attn_arr)

    return {
        'p_har': p_har, 'p_price': p_price, 'p_concat': p_concat, 'p_dual': p_dual,
        'losses_har': losses_har, 'losses_price': losses_price, 'losses_concat': losses_concat, 'losses_dual': losses_dual,
        'per_har': per_har, 'per_price': per_price, 'per_concat': per_concat, 'per_dual': per_dual,
        'cov_dual': cov_dual, 'cov_price': cov_price, 'cov_concat': cov_concat, 'cov_har': cov_har,
        'Y_test': Y_test, 'attn': attn_arr
    }

# ---------------------- quick screening and full stage orchestration ----------------------
def run_quick_stage(past_lens, cfg, model_budget=None):
    quick_summary = []
    for past_len in past_lens:
        out_dir = os.path.join(ROOT_RESULTS, f"pastlen_{past_len}")
        os.makedirs(out_dir, exist_ok=True)
        df = load_consolidated("data/consolidated.csv")
        Xv, Xs, Y, idx_map = prepare_pooled_dataset(df, past_len=past_len, horizons=cfg['horizons'])
        N = len(Xv)
        n_folds = cfg['n_folds']
        fold_sizes = [N // n_folds] * n_folds
        for i in range(N % n_folds):
            fold_sizes[i] += 1
        acc = 0
        fold_results = []
        base_model_params = {
            'price': {'hidden_dim':16, 'd_model':32},
            'concat': {'hidden_dim':16, 'd_model':32},
            'dual': {'vol_hidden':16, 'sent_hidden':16, 'd_model':32, 'n_heads':2}
        }
        if model_budget is not None:
            matched = match_param_budget(model_budget, base_model_params)
            base_model_params.update(matched)

        for fid, fs in enumerate(fold_sizes):
            s = acc; e = acc + fs; acc = e
            if s < max(50, past_len): continue
            res = run_fold(Xv[:s], Xs[:s], Y[:s], Xv[s:e], Xs[s:e], Y[s:e], cfg, base_model_params, quick=True, save_attn=False, out_dir=out_dir)
            fold_results.append(res)
        if len(fold_results) == 0:
            pooled_dual = np.inf
        else:
            pooled_dual = np.mean([res['losses_dual'].mean() for res in fold_results])
        quick_summary.append({'past_len': past_len, 'pooled_dual_quick': pooled_dual, 'n_folds': len(fold_results)})
    quick_sorted = sorted(quick_summary, key=lambda x: x['pooled_dual_quick'])
    chosen = [r['past_len'] for r in quick_sorted[:cfg['top_k']]]
    pd.DataFrame(quick_summary).to_csv(os.path.join(ROOT_RESULTS, "quick_stage_summary.csv"), index=False)
    return quick_summary, chosen

def run_full_stage(chosen_past_lens, cfg, equalize_params=False):
    full_summary = []
    for past_len in chosen_past_lens:
        out_dir = os.path.join(ROOT_RESULTS, f"pastlen_{past_len}")
        os.makedirs(out_dir, exist_ok=True)
        df = load_consolidated("data/consolidated.csv")
        Xv, Xs, Y, idx_map = prepare_pooled_dataset(df, past_len=past_len, horizons=cfg['horizons'])
        N = len(Xv)
        n_folds = cfg['n_folds']
        fold_sizes = [N // n_folds] * n_folds
        for i in range(N % n_folds):
            fold_sizes[i] += 1
        acc = 0
        fold_results = []
        model_params = {
            'price': {'hidden_dim':32, 'd_model':64},
            'concat': {'hidden_dim':32, 'd_model':64},
            'dual': {'vol_hidden':32, 'sent_hidden':32, 'd_model':64, 'n_heads':4}
        }
        if equalize_params:
            model_budget = count_parameters(model_params['dual'], kind='dual')
            matched = match_param_budget(model_budget, model_params)
            model_params.update(matched)
            save_model_size_report(model_params, out_file=os.path.join(out_dir, "model_size_report.json"))

        for fid, fs in enumerate(fold_sizes):
            s = acc; e = acc + fs; acc = e
            if s < max(50, past_len): continue
            fold_out = os.path.join(out_dir, f"fold{fid+1}")
            os.makedirs(fold_out, exist_ok=True)
            res = run_fold(Xv[:s], Xs[:s], Y[:s], Xv[s:e], Xs[s:e], Y[s:e], cfg, model_params, quick=False, save_attn=True, out_dir=fold_out)
            fold_results.append(res)
            def save_mat(mat, fname):
                dfm = pd.DataFrame(mat, index=[f"q={q}" for q in cfg['quantiles']], columns=[f"H={h}" for h in cfg['horizons']])
                dfm.to_csv(os.path.join(fold_out, fname))
            save_mat(res['losses_dual'], "dual_loss_matrix.csv")
            save_mat(res['losses_price'], "price_loss_matrix.csv")
            save_mat(res['losses_har'], "har_loss_matrix.csv")
            pd.DataFrame(res['cov_dual'], index=[f"q={q}" for q in cfg['quantiles']], columns=[f"H={h}" for h in cfg['horizons']]).to_csv(os.path.join(fold_out, "dual_coverage.csv"))

            def per_sample_agg(preds, targ, quantiles):
                Np = preds.shape[0]; agg = np.zeros(Np)
                qarr = np.array(quantiles)
                for i in range(Np):
                    err = np.maximum((qarr - 1) * (targ[i][:,None] - preds[i]) , qarr * (targ[i][:,None] - preds[i]))
                    agg[i] = err.mean()
                return agg

            agg_dual = per_sample_agg(res['p_dual'], res['Y_test'], cfg['quantiles'])
            agg_price = per_sample_agg(res['p_price'], res['Y_test'], cfg['quantiles'])
            agg_concat = per_sample_agg(res['p_concat'], res['Y_test'], cfg['quantiles'])
            agg_har = per_sample_agg(res['p_har'], res['Y_test'], cfg['quantiles'])

            dm_dp_stat, dm_dp_p = diebold_mariano(agg_dual, agg_price, h=1)
            dm_dc_stat, dm_dc_p = diebold_mariano(agg_dual, agg_concat, h=1)
            dm_dh_stat, dm_dh_p = diebold_mariano(agg_dual, agg_har, h=1)
            pd.DataFrame([{'dm_dp': dm_dp_stat, 'dm_pval_dp': dm_dp_p, 'dm_dc': dm_dc_stat, 'dm_pval_dc': dm_dc_p, 'dm_dh': dm_dh_stat, 'dm_pval_dh': dm_dh_p}]).to_csv(os.path.join(fold_out, "dm_stats.csv"), index=False)

        agg = aggregate_and_save_results(fold_results, past_len, cfg, out_dir)
        full_summary.append({'past_len': past_len, 'pooled_dual': float(agg['loss_dual_mean'].mean())})

        # per-asset pooled loss & boxplot
        _, _, _, idx_map_all = prepare_pooled_dataset(load_consolidated("data/consolidated.csv"), past_len=past_len, horizons=cfg['horizons'])
        sample_assets = [a for (d,a) in idx_map_all]
        per_dual_all = np.concatenate([res['per_dual'] for res in fold_results], axis=0) if len(fold_results) > 0 else np.array([])
        assets = sorted(set(sample_assets))
        per_asset_pooled = {}
        aligned_assets = sample_assets[:per_dual_all.shape[0]] if per_dual_all.size else []
        for asset in assets:
            idxs = [i for i,a in enumerate(aligned_assets) if a == asset]
            if len(idxs) == 0: continue
            # compute scalar pooled mean over samples, horizons and quantiles
            vals = per_dual_all[idxs].mean()  # overall scalar
            per_asset_pooled[asset] = float(vals)
        pd.DataFrame(list(per_asset_pooled.items()), columns=['asset','pooled_dual']).to_csv(os.path.join(out_dir, "per_asset_pooled_dual.csv"), index=False)
        if per_dual_all.size:
            plt.figure(figsize=(6,4))
            plt.boxplot([per_dual_all[[i for i,a in enumerate(aligned_assets) if a==asset]].mean(axis=(1,2)) for asset in assets], labels=assets)
            plt.ylabel("per-sample aggregated pinball loss (dual)")
            plt.title(f"Per-asset pooled loss (past_len={past_len})")
            plt.savefig(os.path.join(out_dir, "per_asset_boxplot_dual.png"), bbox_inches='tight')
            plt.close()
        if per_dual_all.size:
            plt.figure(figsize=(6,4))
            # compute list of arrays for each asset
            box_data = [per_dual_all[[i for i,a in enumerate(aligned_assets) if a==asset]].mean(axis=(1,2)) for asset in assets]
            # use tick_labels kwarg (Matplotlib >=3.9)
            try:
                plt.boxplot(box_data, tick_labels=assets)
            except TypeError:
            # older Matplotlib — fallback to labels
                plt.boxplot(box_data, labels=assets)
            plt.ylabel("per-sample aggregated pinball loss (dual)")
            plt.title(f"Per-asset pooled loss (past_len={past_len})")
            plt.savefig(os.path.join(out_dir, "per_asset_boxplot_dual.png"), bbox_inches='tight')
            plt.close()


        # event attention visualization
        df_wide = pd.read_csv("data/dataset.csv", parse_dates=["Date"])
        event_dates = select_event_dates_by_vol_spike(df_wide, top_k=5, method='market_avg')
        attn_list = []
        for fid in range(len(fold_results)):
            fold_file = os.path.join(out_dir, f"fold{fid+1}", "attn_fold.npz")
            if os.path.exists(fold_file):
                arr = np.load(fold_file)['attn']
                attn_list.append(arr)
        if len(attn_list) > 0:
            attn_all = np.concatenate(attn_list, axis=0)
            _, _, _, idx_map_all2 = prepare_pooled_dataset(load_consolidated("data/consolidated.csv"), past_len=past_len, horizons=cfg['horizons'])
            sample_dates = np.array([d for (d,a) in idx_map_all2])[:attn_all.shape[0]]
            for ev in event_dates:
                plot_event_attention_series(attn_all, sample_dates, ev, out_dir=out_dir, window_samples=40)

    pd.DataFrame(full_summary).to_csv(os.path.join(ROOT_RESULTS, "full_stage_summary.csv"), index=False)
    print("Full stage complete. Results in", ROOT_RESULTS)

# ---------------------- aggregation helper ----------------------
def aggregate_and_save_results(fold_results, past_len, cfg, out_dir):
    Q = len(cfg['quantiles']); H = len(cfg['horizons'])
    n_folds = len(fold_results)
    loss_dual_mean = np.zeros((Q, H)); loss_price_mean = np.zeros((Q, H)); loss_concat_mean = np.zeros((Q, H)); loss_har_mean = np.zeros((Q, H))
    cov_dual_mean = np.zeros((Q, H))
    per_diffs_price = []; per_diffs_concat = []; per_diffs_har = []
    for res in fold_results:
        loss_dual_mean += res['losses_dual']
        loss_price_mean += res['losses_price']
        loss_concat_mean += res['losses_concat']
        loss_har_mean += res['losses_har']
        cov_dual_mean += res['cov_dual']
        per_diffs_price.append(res['per_dual'] - res['per_price'])
        per_diffs_concat.append(res['per_dual'] - res['per_concat'])
        per_diffs_har.append(res['per_dual'] - res['per_har'])
    loss_dual_mean /= max(1, n_folds); loss_price_mean /= max(1, n_folds); loss_concat_mean /= max(1, n_folds); loss_har_mean /= max(1, n_folds); cov_dual_mean /= max(1, n_folds)

    QH_rows = []
    for qidx in range(Q):
        for hidx in range(H):
            series_price = np.concatenate([arr[:, hidx, qidx].ravel() for arr in per_diffs_price]) if per_diffs_price else np.array([])
            ci_p, _ = block_bootstrap_metric(series_price, block_size=cfg['block_size'], n_boot=cfg['bootstrap_n']) if series_price.size else (np.nan, (np.nan, np.nan))
            pval_p, _ = bootstrap_p_value(series_price, block_size=cfg['block_size'], n_boot=cfg['bootstrap_n']) if series_price.size else (np.nan, np.array([]))
            series_concat = np.concatenate([arr[:, hidx, qidx].ravel() for arr in per_diffs_concat]) if per_diffs_concat else np.array([])
            ci_c, _ = block_bootstrap_metric(series_concat, block_size=cfg['block_size'], n_boot=cfg['bootstrap_n']) if series_concat.size else (np.nan, (np.nan, np.nan))
            pval_c, _ = bootstrap_p_value(series_concat, block_size=cfg['block_size'], n_boot=cfg['bootstrap_n']) if series_concat.size else (np.nan, np.array([]))
            series_har = np.concatenate([arr[:, hidx, qidx].ravel() for arr in per_diffs_har]) if per_diffs_har else np.array([])
            ci_h, _ = block_bootstrap_metric(series_har, block_size=cfg['block_size'], n_boot=cfg['bootstrap_n']) if series_har.size else (np.nan, (np.nan, np.nan))
            pval_h, _ = bootstrap_p_value(series_har, block_size=cfg['block_size'], n_boot=cfg['bootstrap_n']) if series_har.size else (np.nan, np.array([]))
            QH_rows.append({
                'past_len': past_len,
                'q': cfg['quantiles'][qidx], 'h': cfg['horizons'][hidx],
                'price_ci_lo': float(ci_p[0]) if not np.isnan(ci_p[0]) else np.nan, 'price_ci_hi': float(ci_p[1]) if not np.isnan(ci_p[1]) else np.nan, 'price_pval': float(pval_p) if not np.isnan(pval_p) else np.nan,
                'concat_ci_lo': float(ci_c[0]) if not np.isnan(ci_c[0]) else np.nan, 'concat_ci_hi': float(ci_c[1]) if not np.isnan(ci_c[1]) else np.nan, 'concat_pval': float(pval_c) if not np.isnan(pval_c) else np.nan,
                'har_ci_lo': float(ci_h[0]) if not np.isnan(ci_h[0]) else np.nan, 'har_ci_hi': float(ci_h[1]) if not np.isnan(ci_h[1]) else np.nan, 'har_pval': float(pval_h) if not np.isnan(pval_h) else np.nan
            })
    pd.DataFrame(loss_dual_mean, index=[f"q={q}" for q in cfg['quantiles']], columns=[f"H={h}" for h in cfg['horizons']]).to_csv(os.path.join(out_dir, "agg_dual_loss_matrix_mean.csv"))
    pd.DataFrame(loss_price_mean, index=[f"q={q}" for q in cfg['quantiles']], columns=[f"H={h}" for h in cfg['horizons']]).to_csv(os.path.join(out_dir, "agg_price_loss_matrix_mean.csv"))
    pd.DataFrame(loss_concat_mean, index=[f"q={q}" for q in cfg['quantiles']], columns=[f"H={h}" for h in cfg['horizons']]).to_csv(os.path.join(out_dir, "agg_concat_loss_matrix_mean.csv"))
    pd.DataFrame(loss_har_mean, index=[f"q={q}" for q in cfg['quantiles']], columns=[f"H={h}" for h in cfg['horizons']]).to_csv(os.path.join(out_dir, "agg_har_loss_matrix_mean.csv"))
    pd.DataFrame(cov_dual_mean, index=[f"q={q}" for q in cfg['quantiles']], columns=[f"H={h}" for h in cfg['horizons']]).to_csv(os.path.join(out_dir, "agg_dual_coverage_mean.csv"))
    pd.DataFrame(QH_rows).to_csv(os.path.join(out_dir, "agg_qh_bootstrap_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8,4))
    im = ax.imshow(loss_dual_mean, aspect='auto', cmap='viridis')
    ax.set_title(f"Dual loss matrix (past_len={past_len})")
    ax.set_xlabel("Horizon"); ax.set_ylabel("Quantile")
    ax.set_xticks(np.arange(len(cfg['horizons']))); ax.set_xticklabels([f"H={h}" for h in cfg['horizons']])
    ax.set_yticks(np.arange(len(cfg['quantiles']))); ax.set_yticklabels([f"q={q}" for q in cfg['quantiles']])
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.savefig(os.path.join(out_dir, "agg_dual_loss_heatmap.png"), bbox_inches='tight')
    plt.close()
    return {'loss_dual_mean': loss_dual_mean, 'loss_price_mean': loss_price_mean, 'loss_concat_mean': loss_concat_mean, 'loss_har_mean': loss_har_mean, 'cov_dual_mean': cov_dual_mean, 'q_h_rows': QH_rows}

# ---------------------- main ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--past-lens', nargs='+', type=int, default=DEFAULT_PAST_LENS)
    parser.add_argument('--full', action='store_true', help='Run full stage (after quick) using full model sizes')
    parser.add_argument('--equalize-params', action='store_true', help='Attempt to equalize concat/dual parameter budgets')
    parser.add_argument('--no-sentiment-control', action='store_true', help='Also run no-sentiment control in full stage (zero PC1/PC2)')
    args = parser.parse_args()

    cfg = BASE_CONFIG.copy()
    PAST_LENS = args.past_lens
    QUICK = not args.full

    print("CONFIG:", json.dumps(cfg, indent=2))
    set_seed(cfg['seed'])

    quick_summary, chosen = run_quick_stage(PAST_LENS, cfg)
    print("Quick stage done. chosen past_lens:", chosen)

    run_full_stage(chosen, cfg, equalize_params=args.equalize_params)

    if args.no_sentiment_control:
        print("Note: no-sentiment control requested; currently requires rerunning full stage with no_sentiment=True per fold. Implement as needed.")

    print("All experiments finished. Results located in", ROOT_RESULTS)

if __name__ == "__main__":
    main()