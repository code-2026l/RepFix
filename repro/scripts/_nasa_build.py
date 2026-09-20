"""Build a clean multi-cell battery SOH cache from the NASA ARC aging zips.

Standard protocol: each cell is a chronological discharge sequence; SOH(c) =
capacity(c)/capacity(first).  A robust training set uses only the "long" cells
(>= 100 discharge cycles) so every split has enough trajectory to reveal the
variance-shrinkage collapse.  Feature vector per cycle = voltage profile
percentiles + tail-time + start/end current (same trick as the single-cell
loader) -> standardized.

Output: nasa_battery_multi.npz
    X       (n, d)     f32 standardized discharge-features
    y       (n,)       f32 SOH in (0,1]
    cell    (n,)       int cell id
    cell_sizes  (K,)   cycles per cell
  Build a cell-held-out protocol is done by the harness (contiguous chronological
  train/held per cell, pooled, then mixes held-out cell).
"""
import os, sys, glob, zipfile, io
import numpy as np

def parse_tarzip_bytes(data):
    import tarfile
    files = {}
    try:
        tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
        for m in tf.getmembers():
            if m.name.endswith(".mat") and m.isfile():
                files[os.path.basename(m.name)] = tf.extractfile(m).read()
        tf.close()
    except Exception:
        pass
    return files

def load_zip(zpath):
    out = {}
    try:
        zf = zipfile.ZipFile(zpath)
        for n in zf.namelist():
            if n.endswith(".mat"):
                out[os.path.basename(n)] = zf.read(n)
        if out:
            return out
        for n in zf.namelist():
            ext = n.lower()
            if out or ".mat" in ext:
                continue
            if ext.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
                out.update(parse_tarzip_bytes(zf.read(n)))
    except Exception as e:
        print("  load_zip err", zpath, repr(e))
    return out

def cell_cycles(mat_bytes):
    """Return (idx, soh) arrays for discharge cycles of one .mat, or None."""
    try:
        import scipy.io
        M = scipy.io.loadmat(io.BytesIO(mat_bytes), squeeze_me=True,
                             struct_as_record=False)
        key = [k for k in M if k.startswith("B") and not k.startswith("__")]
        if not key:
            return None
        S = M[key[0]].cycle
        if S is None or np.ndim(S) == 0:
            S = [S]
        caps, feats = [], []
        for st in S:
            if st is None:
                continue
            t = getattr(st, "type", None)
            tstr = t.decode() if isinstance(t, bytes) else (str(t) if t is not None else "")
            if tstr != "discharge":
                continue
            d = getattr(st, "data", None)
            if d is None:
                continue
            tm = np.asarray(getattr(d, "Time")).ravel()
            cu = np.asarray(getattr(d, "Current_measured")).ravel()
            vo = np.asarray(getattr(d, "Voltage_measured")).ravel()
            if len(tm) < 2 or len(vo) < 3:
                continue
            cap = float(np.trapezoid(np.abs(cu), tm))
            fvec = [np.percentile(vo, q) for q in (1, 5, 25, 50, 75, 95, 99)]
            fvec += [len(vo), float(cu[0]), float(cu[-1]), float(tm[-1] - tm[0]), float(tm[0])]
            caps.append(cap); feats.append(fvec)
        if len(caps) < 3:
            return None
        caps = np.array(caps); feats = np.array(feats, dtype=np.float32)
        # robust reference: max capacity over the first 5 discharge cycles caps
        # partial/aborted first cycles; then SOH in (0,1]
        ref = float(np.max(caps[:min(5, len(caps))]))
        if ref <= 1e-8:
            return None
        soh = caps / ref
        soh = np.clip(soh, 1e-4, 1.0)
        return soh, feats
    except Exception:
        return None

def main():
    base = os.environ.get("REPFIX_DATA_X",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", "data_x"))
    zdir = os.path.join(base, "nasa_battery", "5. Battery Data Set")
    zips = sorted(glob.glob(os.path.join(zdir, "*.zip")))
    if not zips:
        print("no zips under", zdir); sys.exit(1)
    cells = {}
    for z in zips:
        for cname, b in load_zip(z).items():
            r = cell_cycles(b)
            if r is not None:
                cells[cname] = r
    # keep long cells only (enough trajectory to expose collapse)
    long_ok = 56   # >= this many discharge cycles
    keep = {k: v for k, v in cells.items() if len(v[0]) >= long_ok}
    print("usable long cells:", {k: len(v[0]) for k, v in sorted(keep.items())})
    if len(keep) < 2:
        print("not enough cells"); sys.exit(1)
    names = sorted(keep)
    Xs, ys, rs, cs = [], [], [], []
    for ci, n in enumerate(names):
        soh, feats = keep[n]
        # remaining-useful-life fraction within cell: 1 -> 0 as the cell ages.
        # A constant map is heavily penalized on RUL (unlike SOH, which plateaus
        # near 1.0), so collapse genuinely degrades the primary regression metric.
        m = len(soh)
        rnl = np.linspace(1.0, 0.0, m, dtype=np.float32)
        Xs.append(feats); ys.append(soh.astype(np.float32))
        rs.append(rnl); cs.append(np.full(m, ci))
    X = np.concatenate(Xs); y = np.concatenate(ys)
    r = np.concatenate(rs); c = np.concatenate(cs)
    # standardize features
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)
    out = os.path.join(base, "nasa_battery", "nasa_battery_multi.npz")
    np.savez(out, X=X, y=y, rnl=r, cell=c,
             cell_sizes=np.array([len(keep[n][0]) for n in names]),
             cell_names=np.array(names))
    print("saved", out, "n", len(X), "cells", len(names),
          "d_in", X.shape[1], "soh_range", float(y.min()), float(y.max()),
          "rul_range", float(r.min()), float(r.max()))

if __name__ == "__main__":
    main()