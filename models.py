import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden_dim * (2 if bidirectional else 1)
    def forward(self, x):
        seq, (hn, cn) = self.lstm(x)
        if self.lstm.bidirectional:
            final = torch.cat([hn[-2], hn[-1]], dim=-1)
        else:
            final = hn[-1]
        return seq, final

class CrossAttentionDecoder(nn.Module):
    """
    Returns both decoder outputs and attention weights.
    Attention weights returned with shape: (B, num_heads, H, Ls)
    """
    def __init__(self, d_model=64, num_horizons=5, n_heads=4, ff_hidden=128, dropout=0.1):
        super().__init__()
        self.pos = nn.Parameter(torch.randn(num_horizons, d_model) * 0.01)
        # set batch_first=True, allow returning per-head weights by setting average_attn_weights=False in forward
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_hidden), nn.ReLU(), nn.Linear(ff_hidden, d_model))
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.num_horizons = num_horizons
        self.n_heads = n_heads
    def forward(self, vol_summary, sent_seq):
        # vol_summary: (B, d) -> queries repeated -> (B, H, d)
        B = vol_summary.shape[0]
        q = vol_summary.unsqueeze(1).repeat(1, self.num_horizons, 1) + self.pos.unsqueeze(0)
        # MultiheadAttention: set need_weights=True, average_attn_weights=False to get per-head weights
        attn_out, attn_weights = self.attn(q, sent_seq, sent_seq, need_weights=True, average_attn_weights=False)
        # attn_weights shape expected: (B, num_heads, H, Ls) if PyTorch supports average_attn_weights=False
        # but some versions return (B * num_heads, H, Ls). Normalize handling:
        if attn_weights.dim() == 3:
            # averaged over heads -> (B, H, Ls)
            attn_weights = attn_weights.unsqueeze(1)  # (B, 1, H, Ls)
            # replicate across heads for compatibility
            attn_weights = attn_weights.repeat(1, self.n_heads, 1, 1)
        elif attn_weights.dim() == 4:
            # expected (B, num_heads, H, Ls)
            pass
        elif attn_weights.dim() == 2:
            # shape (H, Ls) or (H*B, Ls) - fall back: try to reshape if possible
            # best-effort fallback: not ideal but avoids crashes on older versions
            attn_weights = attn_weights.reshape(B, self.n_heads, self.num_horizons, -1)
        x = self.ln1(q + attn_out)
        x = self.ln2(x + self.ff(x))
        return x, attn_weights  # (B, H, d), (B, heads, H, Ls)

class QuantileHead(nn.Module):
    def __init__(self, in_dim, num_quantiles=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, num_quantiles))
    def forward(self, x):
        return self.net(x)  # (B, H, Q)

class PriceOnlyLSTM(nn.Module):
    def __init__(self, vol_input_dim=1, hidden_dim=32, d_model=64, num_horizons=5, quantiles=(0.1,0.5,0.9)):
        super().__init__()
        self.lstm = LSTMEncoder(vol_input_dim, hidden_dim)
        self.proj = nn.Linear(self.lstm.out_dim, d_model)
        self.head = QuantileHead(d_model, num_quantiles=len(quantiles))
        self.num_horizons = num_horizons
    def forward(self, vol_seq):
        _, final = self.lstm(vol_seq)
        z = self.proj(final)
        z = z.unsqueeze(1).repeat(1, self.num_horizons, 1)
        preds = self.head(z)
        return preds

class ConcatLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, d_model=64, num_horizons=5, quantiles=(0.1,0.5,0.9)):
        super().__init__()
        self.lstm = LSTMEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.proj = nn.Linear(self.lstm.out_dim, d_model)
        self.head = QuantileHead(d_model, num_quantiles=len(quantiles))
        self.num_horizons = num_horizons
    def forward(self, X):
        _, final = self.lstm(X)
        summary = self.proj(final).unsqueeze(1).repeat(1, self.num_horizons, 1)
        return self.head(summary)

class DualEncoderModel(nn.Module):
    def __init__(self, vol_input_dim=1, sent_input_dim=2, vol_hidden=32, sent_hidden=32, d_model=64,
                 num_horizons=5, quantiles=(0.1,0.5,0.9), n_heads=4):
        super().__init__()
        self.vol_enc = LSTMEncoder(vol_input_dim, vol_hidden)
        self.sent_enc = LSTMEncoder(sent_input_dim, sent_hidden)
        self.vol_proj = nn.Linear(self.vol_enc.out_dim, d_model)
        self.sent_proj = nn.Linear(self.sent_enc.out_dim, d_model)
        self.cross = CrossAttentionDecoder(d_model=d_model, num_horizons=num_horizons, n_heads=n_heads)
        self.head = QuantileHead(d_model, num_quantiles=len(quantiles))
        self.num_horizons = num_horizons
    def forward(self, vol_seq, sent_seq, return_attentions=False):
        sent_seq_enc, _ = self.sent_enc(sent_seq)       # (B, Ls, sent_out)
        sent_seq_proj = self.sent_proj(sent_seq_enc)    # (B, Ls, d_model)
        _, vol_final = self.vol_enc(vol_seq)
        vol_summary = self.vol_proj(vol_final)          # (B, d_model)
        dec, attn = self.cross(vol_summary, sent_seq_proj)    # (B, H, d), (B, heads, H, Ls)
        preds = self.head(dec)
        if return_attentions:
            return preds, attn
        else:
            return preds