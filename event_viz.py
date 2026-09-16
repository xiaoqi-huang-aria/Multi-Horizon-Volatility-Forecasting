import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from viz import plot_attention_heatmap

def select_event_dates_by_vol_spike(df_wide, top_k=5, asset_col="SPY_RV_log", method="market_avg"):
    """
    df_wide: raw wide CSV (Date + per-asset RV_log columns)
    method:
      - 'market_avg': compute average log-RV across assets each date (mean of all *_RV_log cols) and pick top_k dates
      - 'asset': use a single asset column asset_col and pick top_k dates by that series
      - 'jump': pick dates with largest day-over-day jumps (abs diff) in market avg
    Returns list of pandas.Timestamp (dates).
    """
    df = df_wide.copy()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df['date'] = pd.to_datetime(df['date'])
    # find log columns if using market_avg
    if method == "market_avg":
        log_cols = [c for c in df.columns if c.endswith("_RV_log")]
        if len(log_cols) == 0:
            raise ValueError("No *_RV_log columns found for market_avg.")
        df['market_avg'] = df[log_cols].mean(axis=1)
        top = df.nlargest(top_k, 'market_avg')['date'].tolist()
        return top
    elif method == "asset":
        if asset_col not in df.columns:
            raise ValueError(f"{asset_col} not in df")
        top = df.nlargest(top_k, asset_col)['date'].tolist()
        return top
    elif method == "jump":
        log_cols = [c for c in df.columns if c.endswith("_RV_log")]
        df['market_avg'] = df[log_cols].mean(axis=1)
        df['diff'] = df['market_avg'].diff().abs()
        top = df.nlargest(top_k, 'diff')['date'].tolist()
        return top
    else:
        raise ValueError("Unknown method")

def find_sample_indices_near_date(sample_dates, event_date, window_samples=50):
    """
    sample_dates: array-like of sample dates (pandas.Timestamp)
    event_date: datetime-like
    Returns indices of samples whose sample_date in [event_date - window, event_date + window]
    """
    sd = np.array(pd.to_datetime(sample_dates))
    ev = pd.to_datetime(event_date)
    window_days = int(window_samples)
    mask = (sd >= ev - np.timedelta64(window_days, 'D')) & (sd <= ev + np.timedelta64(window_days, 'D'))
    idxs = np.where(mask)[0]
    return idxs

def plot_event_attention_series(attn_all, sample_dates, event_date, out_dir=".", window_samples=30, head_avg=True):
    """
    attn_all: (N_samples, heads, H, Ls)
    sample_dates: np.array of sample dates aligned to attn_all (len N_samples)
    event_date: date to center on (datetime or string)
    window_samples: how many days before/after (in sample-date units) to include
    head_avg: whether to average heads
    Saves:
      - event_attn_YYYY-MM-DD.png : avg attention heatmap
      - event_attn_timeseries_YYYY-MM-DD.png : attention mass time series around event
    """
    os.makedirs(out_dir, exist_ok=True)
    ev = pd.to_datetime(event_date)
    idxs = find_sample_indices_near_date(sample_dates, ev, window_samples=window_samples)
    if len(idxs) == 0:
        print(f"No samples near event {event_date}")
        return None
    sub = attn_all[idxs]  # (K, heads, H, Ls)
    avg = sub.mean(axis=0)  # (heads, H, Ls)

    # save avg heatmap using shared helper
    heatpath = os.path.join(out_dir, f"event_attn_{ev.date()}.png")
    plot_attention_heatmap(avg, savepath=heatpath, title=f"Attention avg around {ev.date()}", head_avg=head_avg)

    # plot attention mass on recent positions over samples
    # per-sample: collapse heads and horizons -> (K, Ls)
    per_sample_map = sub.mean(axis=1).mean(axis=1)  # (K, Ls)
    recent_k = min(10, per_sample_map.shape[1])
    mass_recent = per_sample_map[:, -recent_k:].sum(axis=1)  # (K,)
    times = pd.to_datetime(sample_dates[idxs])
    plt.figure(figsize=(8,3))
    plt.plot(times, mass_recent, marker='o')
    plt.title(f"Attention mass on last {recent_k} sentiment steps around {ev.date()}")
    plt.xlabel("sample date")
    plt.ylabel("attention mass")
    plt.xticks(rotation=45)
    plt.tight_layout()
    timeseries_path = os.path.join(out_dir, f"event_attn_timeseries_{ev.date()}.png")
    plt.savefig(timeseries_path)
    plt.close()
    return heatpath, timeseries_path