import numpy as np
import pandas as pd

def load_consolidated(path="data/consolidated.csv"):
    """
    Load consolidated long-format CSV produced by data_prep.py.
    Accepts 'Date' or 'date' column names and normalizes to lowercase 'date' in the returned DataFrame.
    Expected columns in file: Date (or date), asset, lrv, pc1, pc2
    """
    df = pd.read_csv(path)
    # accept 'Date' or 'date'
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    elif "date" in df.columns:
        pass
    else:
        # if neither present, try case-insensitive match
        matched = None
        for c in df.columns:
            if c.lower() == "date":
                matched = c
                break
        if matched:
            df = df.rename(columns={matched: "date"})
        else:
            raise KeyError("No date column found in consolidated CSV. Expected 'Date' or 'date'.")
    # parse date column
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isna().any():
        raise ValueError("Some dates in consolidated CSV could not be parsed. Check the 'Date' format.")
    df = df.sort_values(["date", "asset"]).reset_index(drop=True)
    required = {"date", "asset", "lrv", "pc1", "pc2"}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Missing required columns in consolidated CSV: {missing}")
    return df

class RollingWindowGenerator:
    """
    For a single asset DataFrame (sorted by date), yields training windows.
    """
    def __init__(self, df, date_col='date', value_col='lrv', pc_cols=('pc1','pc2'),
                 past_len=60, past_len_sent=None, horizons=(1,2,3,5,10)):
        self.df = df.sort_values(date_col).reset_index(drop=True)
        self.past_len = past_len
        self.past_len_sent = past_len_sent or past_len
        self.horizons = list(horizons)
        self.pc_cols = list(pc_cols)
        self.value_col = value_col

    def generate(self):
        n = len(self.df)
        max_h = max(self.horizons)
        last_start = n - max(self.past_len, self.past_len_sent) - max_h
        for start in range(0, last_start + 1):
            vol_past = self.df.loc[start:start + self.past_len - 1, self.value_col].values.astype(np.float32)
            sent_past = self.df.loc[start:start + self.past_len_sent - 1, self.pc_cols].values.astype(np.float32)
            target_start = start + max(self.past_len, self.past_len_sent)
            targets = []
            for h in self.horizons:
                targets.append(self.df.loc[target_start + h - 1, self.value_col])
            yield vol_past.reshape(self.past_len, 1), sent_past, np.array(targets, dtype=np.float32)

def prepare_pooled_dataset(df, past_len=60, horizons=(1,2,3,5,10)):
    """
    Produce pooled (across assets) arrays:
      - X_vols: (N, past_len, 1)
      - X_sents: (N, past_len, 2)
      - Ys: (N, num_horizons)
      - idx_to_date_asset: list of tuples (date, asset) for each sample (useful for alignment)
    """
    X_vols, X_sents, Ys = [], [], []
    idx_map = []
    for asset, g in df.groupby('asset'):
        g = g.sort_values('date').reset_index(drop=True)
        gen = RollingWindowGenerator(g, past_len=past_len, horizons=horizons)
        dates = g['date'].reset_index(drop=True)
        for i, (vol_past, sent_past, targets) in enumerate(gen.generate()):
            start = i
            sample_date = dates[start + past_len - 1]
            X_vols.append(vol_past)
            X_sents.append(sent_past)
            Ys.append(targets)
            idx_map.append((pd.to_datetime(sample_date), asset))
    if len(X_vols) == 0:
        raise ValueError("No rolling-window samples were generated. Check 'past_len' vs data length and that consolidated.csv has enough rows.")
    return np.stack(X_vols), np.stack(X_sents), np.stack(Ys), idx_map