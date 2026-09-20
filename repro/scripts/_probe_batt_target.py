"""Probe: which battery target / subset makes a constant map (collapse) costly,
so STF recovery is measurable.  Empirically scan several label choices."""
import numpy as np
import cross_domain_mtl as C

def load_raw():
    d = np.load(C._path("nasa_battery", "nasa_battery_multi.npz"))
    return {k: d[k] for k in d.files}

def rmse_of_constant(y):
    """RMSE of predicting the data mean = std of y (assuming normalized var=var==mse of mean pred)."""
    return float(np.sqrt(np.mean((y - y.mean()) ** 2)))

def probe():
    r = load_raw()
    X, cell = r["X"], r["cell"]
    soh = r["y"].astype(np.float32)
    rnl = r["rnl"].astype(np.float32)
    n_cells = int(cell.max()) + 1
    # candidate targets / subsets
    cands = []
    # 1) pooled SOH
    cands.append(("soh_ALL", soh))
    # 2) deep-degrading cells only (SOH reach < 0.5)
    deep = [c for c in range(n_cells) if soh[cell == c].min() < 0.55]
    m = np.isin(cell, deep)
    cands.append(("soh_DEEP", soh[m]))
    # 3) log capacity (compresses 1->0.003 huge range)
    cands.append(("log_soh_ALL", np.log(soh + 0.02)))
    # 4) scaled SOH within cell [0,1]
    sc = np.zeros_like(soh)
    for c in range(n_cells):
        yc = soh[cell == c]
        sc[cell == c] = (yc - yc.min()) / (yc.max() - yc.min() + 1e-8)
    cands.append(("soh_scaled_in_cell", sc))
    # 5) 1 - soh (capacity loss)
    cands.append(("loss_ALL", 1.0 - soh))
    for name, y in cands:
        print(f"{name:20s} n={len(y):5d} mean-pred RMSE={rmse_of_constant(y):.4f} "
              f"entropy-ish std={np.std(y):.4f}")

if __name__ == "__main__":
    probe()