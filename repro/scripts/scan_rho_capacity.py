"""Calibration: predicted critical alpha_c(m) = 1/rho_init(m) vs width m.
Larger m -> larger alpha_c (more robust) would confirm the order parameter
predicts the capacity law. Reports raw stats; the noise is informative.
"""
import json, numpy as np
import collapse_phase as cp

def scan(ms=(8, 16, 32, 64, 128, 256), seeds=8):
    out = {}
    for m in ms:
        rhos = []
        for s in range(seeds):
            Xtr, Ytr = cp.init_data(s)
            model = cp.CollapseModel(16, m, 2).to(Xtr.device)
            r = cp.measure_rho(model, Xtr, Ytr)
            rhos.append(r["rho_proj"])
        out[str(m)] = dict(
            rho_init_mean=float(np.mean(rhos)),
            rho_init_std=float(np.std(rhos)),
            alpha_c_pred=float(1.0 / np.mean(rhos)),
            rho_init_all=rhos,
        )
        print(f"m={m:3d} rho_init={np.mean(rhos):6.3f}+-{np.std(rhos):.3f} "
              f"pred alpha_c={1.0/np.mean(rhos):7.3f}", flush=True)
    with open("../results/th_rho_scan.json", "w") as f:
        json.dump(out, f, indent=2)
    print("saved ../results/th_rho_scan.json")

if __name__ == "__main__":
    scan()