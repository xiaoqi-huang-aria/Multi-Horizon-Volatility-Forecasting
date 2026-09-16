import numpy as np
import statsmodels.api as sm

def build_har_features_from_past_series(past_series, windows=(1,5,22)):
    """
    past_series: 1D array with oldest..recent (length >= max(windows))
    returns: 1D feature vector with averages for each window in `windows`.
    """
    feats = []
    L = len(past_series)
    for w in windows:
        if w == 1:
            feats.append(past_series[-1])
        else:
            if w <= L:
                feats.append(past_series[-w:].mean())
            else:
                # if insufficient history, fallback to mean of available
                feats.append(past_series.mean())
    return np.asarray(feats, dtype=np.float32)

class HARQuantileBaseline:
    """
    HAR-RV baseline using features built from past LRV windows and statsmodels QuantReg.
    - Fits a separate quantile regression per horizon and per quantile level.
    - Fits across pooled training samples (assets × time).
    """
    def __init__(self, windows=(1,5,22)):
        self.windows = tuple(windows)
        self.models = {}  # keys: (h_idx, q) -> fitted model results

    def _featurize_all(self, Xv):
        # Xv: (N, past_len, 1) or (N, past_len)
        if Xv.ndim == 3:
            Xv = Xv[:, :, 0]
        N = Xv.shape[0]
        feat_mat = np.zeros((N, len(self.windows)), dtype=np.float32)
        for i in range(N):
            feat_mat[i] = build_har_features_from_past_series(Xv[i], windows=self.windows)
        return feat_mat  # (N, n_feats)

    def fit(self, Xv_train, Y_train, quantiles=[0.1, 0.5, 0.9]):
        """
        Xv_train: (N_train, past_len, 1)
        Y_train: (N_train, H) target log-RV at each horizon
        quantiles: list of quantile levels
        """
        X = self._featurize_all(Xv_train)  # (N, n_feats)
        X_const = sm.add_constant(X)  # intercept
        H = Y_train.shape[1]
        self.models = {}
        for h in range(H):
            y = Y_train[:, h]
            for q in quantiles:
                # statsmodels QuantReg expects floats
                try:
                    res = sm.QuantReg(y, X_const).fit(q=q, maxiter=1000)
                except Exception as e:
                    # fallback: try with smaller maxiter or different solver if issue arises
                    res = sm.QuantReg(y, X_const).fit(q=q, maxiter=500)
                self.models[(h, float(q))] = res
        return self

    def predict(self, Xv):
        """
        Xv: (N, past_len, 1)
        Returns: preds shape (N, H, Q) in the same quantile order as models were fit.
        """
        X = self._featurize_all(Xv)
        X_const = sm.add_constant(X)
        # collect quantiles and H from learned models
        keys = sorted(self.models.keys(), key=lambda k: (k[0], k[1]))
        if not keys:
            raise ValueError("Model not fitted yet.")
        H = max(k[0] for k in keys) + 1
        q_levels = sorted({k[1] for k in keys})
        Q = len(q_levels)
        N = X.shape[0]
        preds = np.zeros((N, H, Q), dtype=np.float32)
        qidx_map = {q: qi for qi, q in enumerate(q_levels)}
        for (h, q), res in self.models.items():
            qi = qidx_map[float(q)]
            preds[:, int(h), qi] = res.predict(X_const)
        return preds, q_levels