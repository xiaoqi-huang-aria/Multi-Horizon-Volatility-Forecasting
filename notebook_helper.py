# Helper fixes for the results_notebook.ipynb plotting error
# Save this file and either `import notebook_helpers` in the notebook
# or copy-paste the functions below into the notebook cells to replace
# the earlier helper definitions.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_csv_if_exists(path, index_col=0):
    """
    Robust CSV loader for matrices saved from the pipeline.
    Many of the pipeline CSVs were saved with the matrix index in the first column,
    so we read with index_col=0 and coerce all data to numeric.
    Returns DataFrame or None if file missing.
    """
    if not os.path.exists(path):
        return None
    # read and treat first column as index (if appropriate)
    try:
        df = pd.read_csv(path, index_col=index_col)
    except Exception:
        # fallback: read without index_col
        df = pd.read_csv(path)
    # Try to coerce all columns to numeric (this will convert any stray strings to NaN)
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    # If coercion produced all-NaN columns (e.g., the file had an extra unnamed index column),
    # attempt to drop those columns and retry with default index treatment
    if df_numeric.isna().all(axis=0).all():
        # fallback: read without index_col and coerce again
        df = pd.read_csv(path)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
    # Drop columns that are entirely NaN (likely non-data columns)
    df_numeric = df_numeric.dropna(axis=1, how='all')
    # If there are rows/cols all-NaN, drop them too
    df_numeric = df_numeric.dropna(axis=0, how='all')
    # If the DataFrame is now empty, return original df (so caller can inspect)
    if df_numeric.shape[0] == 0 or df_numeric.shape[1] == 0:
        # return original loaded df so the notebook can show it for debugging
        return df
    # Otherwise return numeric DataFrame
    return df_numeric

def plot_heatmap_from_df(df, title=None, xticklabels=None, yticklabels=None):
    """
    Plot a heatmap from a numeric DataFrame.
    df should be numeric (pandas DataFrame). If not, the function will try to coerce.
    """
    # If df is not numeric, attempt coercion
    if not np.issubdtype(df.values.dtype, np.number):
        df = df.apply(pd.to_numeric, errors='coerce')
    if df.isna().all(axis=None):
        raise ValueError("DataFrame contains no numeric data to plot.")
    arr = df.values.astype(float)
    fig, ax = plt.subplots(figsize=(8,4))
    im = ax.imshow(arr, aspect='auto')
    ax.set_title(title or "")
    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=45)
    else:
        # try to use column names if available
        try:
            ax.set_xticks(np.arange(arr.shape[1]))
            ax.set_xticklabels(df.columns.tolist(), rotation=45)
        except Exception:
            pass
    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)
    else:
        try:
            ax.set_yticks(np.arange(arr.shape[0]))
            ax.set_yticklabels(df.index.tolist())
        except Exception:
            pass
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.show()
    return fig

# Quick debug helpers you can run in a notebook cell to inspect the CSV
def inspect_csv(path):
    """
    Load CSV in several ways and print dtypes and small previews to help debug object dtype issues.
    """
    print("Path:", path)
    if not os.path.exists(path):
        print("File not found.")
        return
    print("\nRead with index_col=0:")
    try:
        df0 = pd.read_csv(path, index_col=0)
        print(df0.dtypes)
        display(df0.head())
    except Exception as e:
        print("Failed:", e)
    print("\nRead without index_col:")
    df1 = pd.read_csv(path)
    print(df1.dtypes)
    display(df1.head())
    print("\nCoercing to numeric (apply pd.to_numeric):")
    df1_num = df1.apply(pd.to_numeric, errors='coerce')
    print(df1_num.dtypes)
    display(df1_num.head())