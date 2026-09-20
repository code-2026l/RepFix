"""RepFix baseline reproduction (t4): DLinear + iTransformer on shipped folds.

Self-contained (numpy + torch only, no scipy/numba) so it is DCU-portable.

Data contract (probed):
  baseline_fold{N}.npz
      train_X (n,1680) f32   train_y (n,) f32
      val_X   (v,1680) f32   val_y   (v,) f32
      test_X  (t,1680) f32   test_y  (t,) f32
  1680 = 60 days x 28 factors; labels are RAW forward-5-day returns (std ~0.068),
  which is why a degenerate (predict-0) model scores mse ~= var(y) ~ 0.005.

Reproduces the DLinear / iTransformer numbers in baseline_results.json: same metric
(rank IC, pooled Spearman), same MSE-on-raw-return objective, same per-fold + summary
JSON layout.

Usage:
  python baseline_repro.py --model dlinear --folds 0 --epochs 5          # smoke
  python baseline_repro.py --model dlinear --folds 1,2,3,4,5,6,7,8,9,10  # full
  REPFIX_DATA_DIR=/path python baseline_repro.py --model both --folds 1,2
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

DATA_DIR = os.environ.get("REPFIX_DATA_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
FOLD_PATTERN = os.path.join(DATA_DIR, "baseline_fold{n}.npz")
SEQ_LEN = 60
FEAT = 28
INPUT_DIM = SEQ_LEN * FEAT  # 1680


# ---------------------------------------------------------------- metrics
def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def compute_rank_ic(pred, y):
    """Pooled Spearman rank correlation (the paper's Rank-IC metric)."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(pred) & np.isfinite(y)
    pred, y = pred[m], y[m]
    if len(pred) < 4 or np.std(pred) < 1e-12:
        return 0.0
    rp = _rank(pred)
    ry = _rank(y)
    rp -= rp.mean()
    ry -= ry.mean()
    denom = np.sqrt((rp ** 2).sum() * (ry ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((rp * ry).sum() / denom)


def compute_mse(pred, y):
    pred = np.asarray(pred, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    m = np.isfinite(pred) & np.isfinite(y)
    return float(np.mean((pred[m] - y[m]) ** 2))


# ---------------------------------------------------------------- models
class DLinear(nn.Module):
    """DLinear (Zeng et al., AAAI'23): trend + seasonal decomposition, linear heads."""

    def __init__(self, seq_len: int = SEQ_LEN, n_feat: int = FEAT, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(seq_len, 1)
        self.seasonal = nn.Linear(seq_len, 1)

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, SEQ_LEN, FEAT).permute(0, 2, 1)          # (B,28,60)
        trend = F.avg_pool1d(x, self.kernel, stride=1,
                             padding=self.kernel // 2)             # (B,28,60)
        seasonal = x - trend
        t = self.trend(trend).squeeze(-1)                          # (B,28)
        s = self.seasonal(seasonal).squeeze(-1)                    # (B,28)
        return (t + s).mean(dim=-1)                                # (B,)


class iTransformerModel(nn.Module):
    """iTransformer (Liu et al., ICLR'24): feature-as-token attention, scalar head."""

    def __init__(self, seq_len: int = SEQ_LEN, n_feat: int = FEAT,
                 d_model: int = 64, n_layers: int = 2, n_head: int = 4):
        super().__init__()
        self.embed = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_head, dim_feedforward=d_model * 2,
            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, SEQ_LEN, FEAT).permute(0, 2, 1)          # (B,28,60) feat-tokens
        z = self.embed(x)
        z = self.encoder(z)
        z = z.mean(dim=1)
        return self.head(z).squeeze(-1)


def make_model(name: str) -> nn.Module:
    if name == "dlinear":
        return DLinear()
    if name == "itransformer":
        return iTransformerModel()
    raise ValueError(name)


# ---------------------------------------------------------------- training
def predict_batches(model, X, device, batch):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
            outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs)


def train_fold(model, d, epochs, batch, device, lr, standardize=True, patience=8):
    Xtr = np.asarray(d["train_X"], dtype=np.float32)
    ytr = np.asarray(d["train_y"], dtype=np.float32)
    Xva = np.asarray(d["val_X"], dtype=np.float32)
    yva = np.asarray(d["val_y"], dtype=np.float32)
    Xte = np.asarray(d["test_X"], dtype=np.float32)
    yte = np.asarray(d["test_y"], dtype=np.float32)

    # Per-feature z-score (train stats). Rank-IC is insensitive to monotone
    # per-feature scaling, and this keeps heteroscedastic raw factors (cz20,
    # vol_ratio, amount_ratio) from blowing up the linear heads' gradients.
    if standardize:
        mu = Xtr.mean(axis=0, keepdims=True)
        sd = Xtr.std(axis=0, keepdims=True) + 1e-6
        Xtr = (Xtr - mu) / sd
        Xva = (Xva - mu) / sd
        Xte = (Xte - mu) / sd

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossfn = nn.MSELoss()
    n = len(Xtr)

    best_ic, best_sd = -999.0, None
    bad_ep = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss = 0.0
        cnt = 0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = torch.as_tensor(Xtr[idx], dtype=torch.float32, device=device)
            yb = torch.as_tensor(ytr[idx], dtype=torch.float32, device=device)
            opt.zero_grad()
            loss = lossfn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.item()) * len(idx)
            cnt += len(idx)
        val_ic = compute_rank_ic(predict_batches(model, Xva, device, batch), yva)
        if val_ic > best_ic + 1e-5:
            best_ic = val_ic
            bad_ep = 0
            best_sd = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_ep += 1
        print(f"    ep {ep + 1:2d}/{epochs} loss={ep_loss / max(cnt, 1):.5f} "
              f"val_ic={val_ic:+.5f}", flush=True)
        if bad_ep >= patience:
            print(f"    [early-stop] {patience} epochs no-improve", flush=True)
            break

    if best_sd is not None:
        model.load_state_dict(best_sd)
    val_ic = best_ic
    va_pred = predict_batches(model, Xva, device, batch)
    te_pred = predict_batches(model, Xte, device, batch)
    mse = compute_mse(va_pred, yva)
    oos_ic = compute_rank_ic(te_pred, yte)
    return val_ic, oos_ic, mse


def run_fold(name, fold, epochs, batch, device, lr, patience):
    t0 = time.time()
    path = FOLD_PATTERN.format(n=fold)
    print(f"[fold {fold}] {name} <- {path}", flush=True)
    d = np.load(path, mmap_mode="r")
    model = make_model(name)
    val_ic, oos_ic, mse = train_fold(model, d, epochs, batch, device, lr,
                                     patience=patience)
    sec = time.time() - t0
    print(f"[fold {fold}] {name} val_ic={val_ic:+.5f} oos_ic={oos_ic:+.5f} "
          f"mse={mse:.5f} sec={sec:.1f}", flush=True)
    return {"fold": fold, "val_ic": val_ic, "oos_ic": oos_ic, "mse": mse, "sec": sec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both",
                    choices=["dlinear", "itransformer", "both"])
    ap.add_argument("--folds", default="1,2,3,4,5,6,7,8,9,10",
                    help="comma-separated fold indices (0-10)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", default="baseline_repro.json")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    models = (["dlinear", "itransformer"] if args.model == "both"
              else [args.model])
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"[baseline_repro] device={device} models={models} folds={folds} "
          f"epochs={args.epochs} batch={args.batch}", flush=True)

    results = {m: [] for m in models}
    for m in models:
        for f in folds:
            r = run_fold(m, f, args.epochs, args.batch, device, args.lr,
                         args.patience)
            results[m].append(r)

    summary = {}
    for m in models:
        rs = results[m]
        summary[m] = {
            "mean_val_ic": float(np.mean([r["val_ic"] for r in rs])),
            "mean_oos_ic": float(np.mean([r["oos_ic"] for r in rs])),
            "n_folds": len(rs),
        }
    out = {"results": results, "summary": summary}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[summary] {json.dumps(summary, indent=2)}")
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()