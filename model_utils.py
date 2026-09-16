"""
Utilities to report parameter counts and to adjust model hidden sizes to approximately match
a target parameter budget so we can run fair capacity comparisons between Concat LSTM and Dual encoder.
This is heuristic — we try common sensible mappings rather than exact symbolic solving.
"""
import math
import json
import torch
import numpy as np

# We estimate parameter counts for model configs using simple formulas consistent with models.py
def count_parameters(params, kind='dual'):
    """
    params: dict describing model configuration.
    kind: 'dual' or 'concat' or 'price'
    Return approximate parameter count (int).
    """
    if kind == 'price':
        # LSTM params: 4 * hidden * (input + hidden + 1) per layer (single layer)
        input_dim = 1
        hidden = params.get('hidden_dim', 32)
        lstm_params = 4 * hidden * (input_dim + hidden + 1)
        proj = hidden * params.get('d_model', 64)
        head = params.get('d_model', 64) * 64  # rough
        total = lstm_params + proj + head
    elif kind == 'concat':
        input_dim = params.get('input_dim', 3)  # vol + 2 pcs
        hidden = params.get('hidden_dim', 32)
        lstm_params = 4 * hidden * (input_dim + hidden + 1)
        proj = hidden * params.get('d_model', 64)
        head = params.get('d_model', 64) * 64
        total = lstm_params + proj + head
    else:  # dual
        # vol LSTM + sent LSTM + projections + cross-attention weights
        vol_h = params.get('vol_hidden', 32)
        sent_h = params.get('sent_hidden', 32)
        d = params.get('d_model', 64)
        n_heads = params.get('n_heads', 4)
        # LSTM params
        vol_lstm = 4 * vol_h * (1 + vol_h + 1)
        sent_lstm = 4 * sent_h * (2 + sent_h + 1)
        # projections
        vol_proj = vol_h * d
        sent_proj = sent_h * d
        # multihead attention params: in_proj (d*d*3) + out_proj (d*d)
        attn = d * d * 4
        head_proj = d * 64
        total = vol_lstm + sent_lstm + vol_proj + sent_proj + attn + head_proj
    return int(total)

def match_param_budget(target_budget, base_params):
    """
    Given a target_budget (int) and base_params dict, return modified params for concat & dual
    that try to match the target budget.
    base_params is expected to contain 'concat'/'dual' default params.
    This is heuristic: we scale hidden dims proportionally.
    """
    # compute current budgets
    cur_dual = count_parameters(base_params.get('dual', {}), kind='dual')
    cur_concat = count_parameters(base_params.get('concat', {}), kind='concat')
    if cur_dual == 0:
        return {}
    scale = math.sqrt(target_budget / cur_dual)
    # scale dual dims
    d = max(8, int(base_params['dual'].get('d_model',64) * scale))
    vol_h = max(8, int(base_params['dual'].get('vol_hidden',32) * scale))
    sent_h = max(8, int(base_params['dual'].get('sent_hidden',32) * scale))
    # scale concat to similar budget
    # invert: find concat hidden so that its estimated params ≈ target_budget
    # brute-force search for hidden between 8..256
    best = None
    best_diff = 1e12
    for h in range(8, 257, 4):
        trial = {'hidden_dim': h, 'd_model': base_params['concat'].get('d_model',64), 'input_dim': base_params['concat'].get('input_dim',3)}
        val = count_parameters(trial, kind='concat')
        diff = abs(val - target_budget)
        if diff < best_diff:
            best_diff = diff; best = (h, val)
    concat_h = best[0]
    matched = {
        'dual': {'vol_hidden': vol_h, 'sent_hidden': sent_h, 'd_model': d, 'n_heads': base_params['dual'].get('n_heads',4)},
        'concat': {'hidden_dim': concat_h, 'd_model': base_params['concat'].get('d_model',64)}
    }
    return matched

def save_model_size_report(model_params, out_file="model_sizes.json"):
    report = {}
    report['price'] = count_parameters(model_params.get('price', {}), kind='price')
    report['concat'] = count_parameters(model_params.get('concat', {}), kind='concat')
    report['dual'] = count_parameters(model_params.get('dual', {}), kind='dual')
    with open(out_file, 'w') as f:
        json.dump(report, f, indent=2)
    return report