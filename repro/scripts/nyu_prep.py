"""NYUv2 (HF tanganke/nyuv2) parquet -> npz converter, half resolution 144x192.

Reads the 14 parquet shards (8 train / 6 val), 2x2-area-downsamples
image/depth/normal (nearest for segmentation, normals re-normalised after
averaging), builds 9x12 coarse grids (depth block mean; segmentation majority
vote ignoring -1=unlabeled, all-invalid cell -> -1; normals block mean +
re-normalise), and writes data_x/nyu/nyu_nyu.npz used by
cross_domain_mtl.py::_load_nyu (X + depth grid) and nyu_bmtl.py (full maps).

Full-dataset facts (v2, after the single-sample gate misfire): segmentation
labels are -1 (unlabeled) + 13 valid classes 0..12; depth spans [0, ~10] m;
some pixels carry zero normals (invalid) -> zero after block averaging, and
the benchmark masks them out.

Sanity gates: exact split sizes 795/654, seg label range [-1, 12], depth in
[0, 10.5] m, NaN check.
"""
import numpy as np
import pyarrow.parquet as pq
import os
import time

_DX = os.environ.get("REPFIX_DATA_X",
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "data_x"))
ROOT = os.path.join(_DX, "nyu", "hf", "data")
OUT = os.path.join(_DX, "nyu", "nyu_nyu.npz")
N_SEG = 13          # valid classes 0..12 (-1 = unlabeled / ignore)


def avg2(a):
    """(..., H, W) -> (..., H/2, W/2) via 2x2 area averaging."""
    h, w = a.shape[-2], a.shape[-1]
    assert h % 2 == 0 and w % 2 == 0, (h, w)
    return a.reshape(*a.shape[:-2], h // 2, 2, w // 2, 2).mean(axis=(-3, -1))


def near2(a):
    return a[..., ::2, ::2]


def dep_grid(d, gh=9, gw=12):
    """(N, 144, 192) -> (N, 9, 12) block mean (16x16 blocks)."""
    n = d.shape[0]
    return d.reshape(n, gh, 16, gw, 16).mean((2, 4)).astype(np.float32)


def nor_grid(n, gh=9, gw=12):
    n_ = n.shape[0]
    g = n.reshape(n_, 3, gh, 16, gw, 16).mean((3, 5))
    nn = np.sqrt((g * g).sum(1, keepdims=True))
    return (g / np.maximum(nn, 1e-6)).astype(np.float32)


def seg_grid(seg, gh=9, gw=12):
    """(N, 144, 192) -> (N, 9, 12) majority label ignoring -1 (all-invalid
    cell -> -1, kept for CE ignore_index)."""
    n = seg.shape[0]
    g = seg.reshape(n, gh, 16, gw, 16)
    out = np.full((n, gh, gw), -1, dtype=np.int8)
    for r in range(gh):
        for c in range(gw):
            lab = g[:, r, :, c, :].reshape(n, -1)
            valid = lab >= 0
            has = valid.any(1)
            cnt = np.zeros((n, N_SEG), dtype=np.int32)
            for v in range(N_SEG):
                cnt[:, v] = ((lab == v) & valid).sum(1)
            maj = cnt.argmax(1).astype(np.int8)
            out[:, r, c] = np.where(has, maj, np.int8(-1))
    return out


def load_split(prefix, nshard):
    Xs, Ss, Ds, Ns = [], [], [], []
    for i in range(nshard):
        t0 = time.time()
        f = os.path.join(ROOT, f"{prefix}-{i:05d}-of-{nshard:05d}.parquet")
        t = pq.ParquetFile(f).read().to_pydict()
        n = len(t["image"])
        for j in range(n):
            img = np.asarray(t["image"][j], dtype=np.float32)
            seg = np.asarray(t["segmentation"][j])
            dep = np.asarray(t["depth"][j], dtype=np.float32)
            nor = np.asarray(t["normal"][j], dtype=np.float32)
            if dep.ndim == 3:
                dep = dep[0]
            assert img.shape == (3, 288, 384), img.shape
            assert seg.shape == (288, 384), seg.shape
            assert nor.shape == (3, 288, 384), nor.shape
            img2 = avg2(img)                       # (3,144,192)
            dep2 = avg2(dep)                       # (144,192)
            nor2 = avg2(nor)                       # (3,144,192)
            nn = np.sqrt((nor2 * nor2).sum(0, keepdims=True))
            nor2 = nor2 / np.maximum(nn, 1e-6)
            seg2 = near2(seg).astype(np.int8)      # (144,192), -1..12
            Xs.append(img2); Ss.append(seg2); Ds.append(dep2); Ns.append(nor2)
        print(f"{prefix} shard {i + 1}/{nshard} done in {time.time() - t0:.1f}s",
              flush=True)
    return np.stack(Xs), np.stack(Ss), np.stack(Ds), np.stack(Ns)


t0 = time.time()
Xtr, Str, Dtr, Ntr = load_split("train", 8)
Xte, Ste, Dte, Nte = load_split("val", 6)
print("loaded", Xtr.shape, Xte.shape, f"in {time.time() - t0:.0f}s", flush=True)

# ---- sanity gates -----------------------------------------------------------
assert Xtr.shape[0] == 795 and Xte.shape[0] == 654, (Xtr.shape, Xte.shape)
assert Xtr.shape[1:] == (3, 144, 192)
for nm, S in (("train", Str), ("val", Ste)):
    print(nm, "seg range", int(S.min()), int(S.max()),
          "frac -1: %.4f" % float((S == -1).mean()))
print("depth train mean %.3f max %.3f" % (float(Dtr.mean()), float(Dtr.max())))
print("normal unit-norm mean %.4f min %.4f"
      % (float(np.linalg.norm(Ntr, axis=1).mean()),
         float(np.linalg.norm(Ntr, axis=1).min())))
assert Str.min() >= -1 and Str.max() <= 12
assert Dtr.min() >= -1e-6 and Dtr.max() <= 10.5
assert not np.isnan(Xtr).any() and not np.isnan(Dtr).any()
assert not np.isnan(Ntr).any() and not np.isnan(Xte).any()

ytr = dep_grid(Dtr)
yte = dep_grid(Dte)
sgtr = seg_grid(Str)
sgte = seg_grid(Ste)
ngtr = nor_grid(Ntr)
ngte = nor_grid(Nte)
print("grids:", ytr.shape, sgtr.shape, ngtr.shape,
      "seg-grid frac -1: %.4f" % float((sgtr == -1).mean()), flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT,
         X_tr=Xtr, seg_tr=Str, dep_tr=Dtr, nor_tr=Ntr,
         y_tr=ytr, sgr_tr=sgtr, ngr_tr=ngtr,
         X_te=Xte, seg_te=Ste, dep_te=Dte, nor_te=Nte,
         y_te=yte, sgr_te=sgte, ngr_te=ngte)
print("wrote %s (%.2f GB) in %.0fs"
      % (OUT, os.path.getsize(OUT) / 1e9, time.time() - t0), flush=True)
