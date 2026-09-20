"""Extract multi-cell ARC battery SOH dataset -> a single npy cache.

Crosses the aging-set zips (BatteryAgingARC_FY08Q4 / B25-28 / B25-44 / B45-48 /
B49-52 / B53-56) and builds a pooled discharge-capacity table with SOH labels.

Output: nasa_battery_big.npz with X (n, d_in) standardized, y (n,) SOH in [0,1],
plus per-cell split groups so a fair cell-held-out protocol is possible.
"""
import os, sys, zipfile, io, glob
import numpy as np

def parse_tarzip_bytes(data):
    """data is bytes of a tar that itself contains .mat files."""
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
    """Return {cell_name: mat_bytes} by trying nested zip/tar structures."""
    out = {}
    try:
        zf = zipfile.ZipFile(zpath)
        for n in zf.namelist():
            if n.endswith(".mat"):
                out[os.path.basename(n)] = zf.read(n)
        if out:
            return out
        # else: single member may be a tar of mats
        for n in zf.namelist():
            ext = n.lower()
            if out or (".mat" in ext):
                continue
            if ext.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
                inner = parse_tarzip_bytes(zf.read(n))
                out.update(inner)
    except Exception as e:
        print("  load_zip err", zpath, repr(e))
    return out

def soh_cells(mat_bytes):
    """Parse one .mat and return (cycle_index[], soh[]) of discharge cycles."""
    try:
        import scipy.io
        bio = io.BytesIO(mat_bytes)
        M = scipy.io.loadmat(bio, squeeze_me=True, struct_as_record=False)
        # find the battery struct key (e.g. 'B0005','B0025'...)
        key = [k for k in M if k.startswith("B") and not k.startswith("__")]
        if not key:
            return None
        S = M[key[0]].cycle
        if S is None or np.ndim(S) == 0:
            S = [S]
        caps = {}
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
            if len(tm) < 2:
                continue
            cap = float(np.trapezoid(np.abs(cu), tm))
            caps[len(caps)] = cap
        if not caps:
            return None
        idx = np.array(list(caps.keys()))
        arr = np.array(list(caps.values()))
        soh = arr / arr[0]
        return idx, soh
    except Exception as e:
        print("  soh err", repr(e))
        return None

def main():
    base = os.environ.get("REPFIX_DATA_X",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "..", "..", "data_x"))
    zdir = os.path.join(base, "nasa_battery", "5. Battery Data Set")
    if not os.path.isdir(zdir):
        zdir = os.path.join(os.getcwd(), "nasa_battery", "5. Battery Data Set")
    zips = sorted(glob.glob(os.path.join(zdir, "*.zip")))
    if not zips:
        print("no zips found under", zdir); sys.exit(1)
    cells = {}
    for z in zips:
        print("zip", os.path.basename(z))
        mats = load_zip(z)
        for cname, b in mats.items():
            r = soh_cells(b)
            if r is not None:
                cells[cname] = r
                print("  ", cname, "cycles", len(r[1]))
            else:
                print("  ", cname, "parse-fail")
    if not cells:
        print("no usable cells"); sys.exit(1)
    return cells

if __name__ == "__main__":
    cells = main()
    out = os.path.join(os.getcwd(), "_cells_probe.json")
    import json
    json.dump({k:[int(v[0][0]), float(v[1][0]), float(v[1][-1])] for k,v in cells.items()},
              open(out,"w"), indent=2)
    print("probe saved", out, "->", {k:len(v[1]) for k,v in cells.items()})