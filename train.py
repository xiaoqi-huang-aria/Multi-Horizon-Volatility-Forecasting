import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import PriceOnlyLSTM, ConcatLSTM  # used to decide how to call model

# deterministic seeds
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_nn(model, train_loader: DataLoader, val_loader: DataLoader, quantiles,
             lr=1e-3, epochs=50, patience=8, device='cpu'):
    """
    Training loop that supports:
      - PriceOnlyLSTM  : receives (vol) input
      - ConcatLSTM     : receives concatenated (vol, sent) input
      - DualEncoderModel: receives (vol, sent) input and returns preds or (preds, attn)

    Expects train_loader/val_loader to yield either:
      - (vol, sent, targ)  OR
      - (vol, targ)        (kept for backward compatibility)
    """
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val = np.inf
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            # support both 2-tuple and 3-tuple batches
            if len(batch) == 3:
                vol, sent, targ = batch
                sent = sent.to(device).float()
            elif len(batch) == 2:
                vol, targ = batch
                sent = None
            else:
                raise ValueError(f"Unexpected batch length {len(batch)} from DataLoader")

            vol = vol.to(device).float()
            targ = targ.to(device).float()

            # forward
            if isinstance(model, PriceOnlyLSTM):
                preds = model(vol)
            elif isinstance(model, ConcatLSTM):
                if sent is None:
                    raise ValueError("ConcatLSTM expects (vol, sent, targ) batches")
                inp = torch.cat([vol, sent], dim=-1)
                preds = model(inp)
            else:
                # assume dual encoder style: (vol, sent)
                if sent is None:
                    raise ValueError("Model expects both vol and sent inputs but batch has no sent.")
                preds = model(vol, sent)

            # compute pinball loss (mean over batch/horizons/quantiles)
            # quantiles is a list-like of Q values
            q = torch.tensor(quantiles, device=preds.device, dtype=preds.dtype)
            targets_exp = targ.unsqueeze(-1).expand_as(preds)
            errors = targets_exp - preds
            loss_tensor = torch.max((q - 1) * errors, q * errors)  # (B, H, Q)
            loss = loss_tensor.mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_losses.append(loss.item())

        # validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    vol, sent, targ = batch
                    sent = sent.to(device).float()
                elif len(batch) == 2:
                    vol, targ = batch
                    sent = None
                else:
                    raise ValueError(f"Unexpected batch length {len(batch)} from DataLoader")
                vol = vol.to(device).float()
                targ = targ.to(device).float()

                if isinstance(model, PriceOnlyLSTM):
                    preds = model(vol)
                elif isinstance(model, ConcatLSTM):
                    if sent is None:
                        raise ValueError("ConcatLSTM expects (vol, sent, targ) batches")
                    inp = torch.cat([vol, sent], dim=-1)
                    preds = model(inp)
                else:
                    if sent is None:
                        raise ValueError("Model expects both vol and sent inputs but batch has no sent.")
                    preds = model(vol, sent)

                q = torch.tensor(quantiles, device=preds.device, dtype=preds.dtype)
                targets_exp = targ.unsqueeze(-1).expand_as(preds)
                errors = targets_exp - preds
                loss_tensor = torch.max((q - 1) * errors, q * errors)
                val_losses.append(loss_tensor.mean().item())

        avg_val = float(np.mean(val_losses)) if val_losses else float('inf')
        # early stopping / checkpointing
        if avg_val < best_val - 1e-8:
            best_val = avg_val
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        # optional: print progress
        print(f"Epoch {epoch+1}/{epochs} train_loss={np.mean(train_losses):.6f} val_loss={avg_val:.6f} wait={wait}")

        if wait >= patience:
            print("Early stopping")
            break

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)
    return model