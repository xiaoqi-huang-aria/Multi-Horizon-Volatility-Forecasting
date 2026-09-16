import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os

INPUT_CSV = "data/dataset.csv"
OUTPUT_CSV = "data/consolidated.csv"

# Config
ROLLING_WINDOW = 120
SENT_COLS = ["EPU", "GPR", "TPU", "FNSI"]  
RV_LOG_SUFFIX = "_RV_log"

def _find_date_column(df):
    for c in ("Date", "date"):
        if c in df.columns:
            return c
    for c in df.columns:
        if c.lower() == "date":
            return c
    raise KeyError("No date column found. Expected 'Date' or 'date' in CSV header.")

def melt_wide_to_long(df_wide, rv_log_suffix=RV_LOG_SUFFIX):
    # find date column
    date_col = _find_date_column(df_wide)
    # find all columns ending with the suffix
    rv_log_cols = [c for c in df_wide.columns if c.endswith(rv_log_suffix)]
    if not rv_log_cols:
        raise ValueError(f"No columns found with suffix '{rv_log_suffix}'. Found columns: {list(df_wide.columns)}")
    # derive asset names
    assets = [c[:-len(rv_log_suffix)] for c in rv_log_cols]
    df_list = []
    for col, asset in zip(rv_log_cols, assets):
        # ensure the sentiment columns exist; if they don't, we'll set NaNs and warn
        cols_needed = [date_col] + [sc for sc in SENT_COLS if sc in df_wide.columns] + [col]
        tmp = df_wide[cols_needed].copy()
        tmp = tmp.rename(columns={col: "lrv"})
        tmp["asset"] = asset
        # if any SENT_COLS were missing, add as NaN columns to keep schema consistent
        for sc in SENT_COLS:
            if sc not in tmp.columns:
                tmp[sc] = np.nan
        # normalize date column name to 'Date' for output
        tmp = tmp.rename(columns={date_col: "Date"})
        tmp["lrv"] = pd.to_numeric(tmp["lrv"], errors="coerce")
        df_list.append(tmp)
    long = pd.concat(df_list, ignore_index=True)
    cols = ["Date", "asset"] + SENT_COLS + ["lrv"]
    # ensure all columns present
    for c in cols:
        if c not in long.columns:
            long[c] = np.nan
    return long[cols]

def compute_pooled_rolling_pcs(df_wide, sent_cols=SENT_COLS, window=ROLLING_WINDOW):
    """
    Compute pooled rolling PCA on sentiment variables across dates.
    Returns a DataFrame with columns Date, pc1, pc2 (NaN for the first `window` dates).
    """
    date_col = _find_date_column(df_wide)
    # build a date-sorted unique date frame that contains sentiment vars
    df_dates = df_wide[[date_col] + [sc for sc in sent_cols if sc in df_wide.columns]].drop_duplicates(subset=[date_col])
    df_dates = df_dates.sort_values(date_col).reset_index(drop=True).copy()
    df_dates = df_dates.rename(columns={date_col: "Date"})
    pcs = np.full((len(df_dates), 2), np.nan, dtype=np.float32)
    # if sentiment cols missing entirely, raise
    present_sent = [sc for sc in sent_cols if sc in df_dates.columns]
    if len(present_sent) == 0:
        raise ValueError(f"No sentiment columns found among expected {sent_cols}. Available columns: {list(df_wide.columns)}")
    for i in range(window, len(df_dates)):
        X = df_dates.loc[i - window:i - 1, present_sent].values.astype(float)
        mu = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0) + 1e-8
        Xs = (X - mu) / sigma
        pca = PCA(n_components=2)
        pca.fit(Xs)
        row = df_dates.loc[i, present_sent].values.astype(float)
        row_s = (row - mu) / sigma
        pc = pca.transform(row_s.reshape(1, -1)).reshape(-1)
        pcs[i, :] = pc
    df_dates["pc1"] = pcs[:, 0]
    df_dates["pc2"] = pcs[:, 1]
    return df_dates[["Date", "pc1", "pc2"]]

def main(in_csv=INPUT_CSV, out_csv=OUTPUT_CSV):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df = pd.read_csv(in_csv)
    # ensure we have a Date column in datetime format in the wide table
    date_col = _find_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if df[date_col].isna().any():
        raise ValueError("Some Date values could not be parsed. Check the input CSV 'Date' formatting.")
    long = melt_wide_to_long(df)
    pcs_by_date = compute_pooled_rolling_pcs(df, sent_cols=SENT_COLS, window=ROLLING_WINDOW)
    # merge pooled PCs onto long table by Date
    merged = pd.merge(long, pcs_by_date, on="Date", how="left")
    # drop rows without pc values (first window) or without lrv
    merged = merged.dropna(subset=["pc1", "pc2", "lrv"]).reset_index(drop=True)
    # save with 'Date' column name (capital D) for clarity
    merged.to_csv(out_csv, index=False)
    print(f"Saved consolidated long file: {out_csv}, rows: {len(merged)}")

if __name__ == "__main__":
    main()