"""battery_preprocess.py — unified cycle-sequence extractor for NASA / MATR / CALCE.

Produce battery_parsed/<ds>_cycles.json (a list of per-cycle dicts) consumed by
battery_bmtl.py:

  cycle dict = {
    "voltage":    [T] or []   per-cycle discharge/charge voltage profile
    "current":    [T]           current profile
    "temperature":[T]           temperature profile
    "resistance": [T]           internal-resistance profile (if available)
    "SOH":        float   capacity retention (%) at this cycle
    "RUL":        int     remaining cycles to end-of-life (80% capacity)
    "C_rate":     float   charge rate (C) of this cycle
    "temp":       float   mean cycle temperature (C)
    "dSOH":       float   dSOH per cycle (for re-weighting degradation term)
  }

Sources:
  NASA : BatteryAgingARC (B00x/B0x0 ...) 19 cells, charge/discharge .mat
  MATR : Severson high-throughput fast-charge .mat (HDF5), 46+ cells/batch
  CALCE: CS2/CX2 .xlsx per-cycle spreadsheets (multi-rate)

Run on the A100 box (raw files absent locally).  All feature vectors are
downsampled to <= MAX_TS points; missing features become [] (model zero-pads).
"""
from __future__ import annotations

import os
R = os.environ.get("REPFIX_RESULTS",
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "results"))

import argparse, json, os, math, re, zipfile, io, glob

import numpy as np
_np = np

ROOT = os.path.join(os.path.dirname(R), "data_x")
RAW = os.path.join(ROOT, "batteryml_raw")
OUT = os.path.join(ROOT, "battery_parsed")
NAS = os.path.join(ROOT, "nasa_battery")
MAX_TS = 256
EOL_SOH = 80.0          # end-of-life capacity threshold (%)


# ------------------------------------------------------------------------- NASA
def load_mat_v7(path):
    import scipy.io
    d = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    return d


def nasa_cycles():
    """Parse NASA .mat files -> list of cycle dicts (one per discharge cycle).

    NASA PCoE format: B['cycle'] is a length-n structured ndarray with fields
    ['ambient_temperature','data','time','type']; each .data is a structured
    array with Capacity/(V/I/T) measured series.  Capacity of discharge cycles
    drives SOH; the charge/discharge voltage-current-temperature sequences are
    kept as the model's cross-sectional features.
    """
    import scipy.io
    cyc = []
    mat_files = sorted(glob.glob(os.path.join(NAS, "*.mat")))
    seen = set()
    for p in mat_files:
        base = os.path.basename(p)
        if base in seen:
            continue
        seen.add(base)
        if base in ("B0018.mat",):          # 263-byte broken stub
            continue
        try:
            d = scipy.io.loadmat(p, struct_as_record=False, squeeze_me=True)
        except Exception as e:
            print("  skip", base, type(e).__name__, e)
            continue
        key = [k for k in d if not k.startswith("__")]
        if not key:
            continue
        B = d[key[0]]
        if not hasattr(B, "cycle"):
            print("  no cycle attr in", base)
            continue
        try:
            n = len(B.cycle)
        except Exception as e:
            n = 1
        if not hasattr(B.cycle, "__getitem__") or (hasattr(n, "__len__")):
            n = 1
        caps = np.zeros(n)
        for i in range(n):
            try:
                cy = B.cycle[i]
                cytype = str(getattr(cy, "type", ""))
                dd = getattr(cy, "data", None)
                if dd is None:
                    caps[i] = np.nan
                    continue
                if "discharge" in cytype or "DC" in cytype:
                    cap = getattr(dd, "Capacity", None)
                    caps[i] = float(np.sum(np.asarray(cap, float))) if cap is not None else np.nan
                else:
                    caps[i] = np.nan
            except Exception:
                caps[i] = np.nan
        nom = nanmax(caps)
        if nom <= 0:
            print("  bad capacity in", base)
            continue
        # collect discharge cycles in order
        disc = [i for i in range(n) if "discharge" in str(getattr(B.cycle[i], "type", ""))]
        for kcur, i in enumerate(disc):
            cy = B.cycle[i]
            dd = getattr(cy, "data", None) or _np.zeros(0)
            def ser(f):
                try:
                    return _f32(getattr(dd, f))
                except Exception:
                    return _np.zeros(0)
            volt = ser("Voltage_measured") if hasattr(dd, "Voltage_measured") else _np.zeros(0)
            curr = ser("Current_load") if hasattr(dd, "Current_load") else _np.zeros(0)
            temp = ser("Temperature_measured") if hasattr(dd, "Temperature_measured") else _np.zeros(0)
            cap = _np.asarray(getattr(dd, "Capacity", _np.nan), float)
            cap = float(np.sum(cap)) if cap.size else np.nan
            soh = 100.0 * cap / nom if np.isfinite(cap) else 0.0
            rul = max(0, int(len(disc) - 1 - kcur))
            cyc.append(dict(
                voltage=volt, current=curr, temperature=temp, resistance=_np.zeros(0),
                SOH=float(soh), RUL=int(rul), C_rate=float(1.0),
                temp=float(np.mean(temp)) if temp.size else 25.0, dSOH=0.0, _cid=base))
    if cyc:
        for i in range(len(cyc) - 1):
            if cyc[i]["SOH"] > 0 and cyc[i + 1]["SOH"] > 0:
                cyc[i]["dSOH"] = max(0.0, cyc[i]["SOH"] - cyc[i + 1]["SOH"])
    print(f"    NASA: {len(cyc)} discharge cycles from {len(seen)} files")
    return cyc


def nanmax(a):
    a = _f32(a)
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else 0.0


def _f32(a):
    return np.asarray(a, np.float32).ravel()


def _norm(a, lo, hi):
    a = _f32(a)
    hi = hi - lo if hi - lo > 1e-9 else 1.0
    return ((a - lo) / hi).astype(np.float32)


# ------------------------------------------------------------------------- MATR
def _win_cycles(summary_arrays, feats_scale, cid, step=8):
    """Turn per-cell per-cycle summary seqs into sliding-window training samples.
    summary_arrays: dict ch -> np.ndarray of per-cycle scalars (Qd, IR, Tavg, ...).
    feats_scale: list of (lo, hi) to normalise each channel.
    Each window becomes a sample: features are the previous `W` cycles of the
    4 summary channels; SOH = last-window capacity normalised to [0,1]; RUL =
    remaining cycles until this cell's Qd crosses its own EOL threshold.
    step: window stride (adjacent windows overlap W-1 cycles; >1 de-redundancies)."""
    n = len(summary_arrays["Qd"])
    if n < 16:
        return []
    qd = np.asarray(summary_arrays["Qd"], float)
    nom = nanmax(qd)
    eol = EOL_SOH / 100.0 * nom
    eol_idx = n  # default: never reached
    hit = np.where(qd <= eol)[0]
    if hit.size:
        eol_idx = int(hit[0])
    W = MAX_TS
    out = []
    pad = min(4, n - 1)
    for i in range(pad, n, step):
        lo = max(0, i - W + 1)
        chans = []
        for ch, (lo_v, hi_v) in zip(("Qd", "IR", "Tavg", "Tmin"), feats_scale):
            seq = np.asarray(summary_arrays.get(ch, np.zeros(n)), float)[lo:i + 1]
            chans.append(_norm(seq, lo_v, hi_v))
        x = np.stack(chans, axis=1)            # (T,4)
        tlen = x.shape[0]
        if tlen < W:
            x = np.concatenate([np.zeros((W - tlen, 4), np.float32), x], 0)
        soh = 100.0 * qd[i] / nom if nom > 0 else 0.0
        rul = max(0, int(eol_idx - i))
        dsoh = max(0.0, qd[i] - qd[i - step]) / nom * 100.0 if i >= step and nom > 0 else 0.0
        out.append(dict(voltage=x[:, 0], current=x[:, 1], temperature=x[:, 2],
                        resistance=x[:, 3], SOH=float(soh), RUL=int(rul),
                        C_rate=float(1.0), temp=float(np.mean(summary_arrays["Tavg"][lo:i + 1])) if "Tavg" in summary_arrays else 25.0,
                        dSOH=float(dsoh), _cid=cid))
    return out


def matr_cycles():
    """Parse MATR HDF5 batches.  Each cell's summary seqs become windows."""
    import h5py
    cyc = []
    for p in sorted(glob.glob(os.path.join(RAW, "MATR", "*.mat"))):
        print("  MATR", os.path.basename(p))
        try:
            d = h5py.File(p, "r")
        except Exception as e:
            print("    skip", type(e).__name__, e)
            continue
        batch = d["batch"]
        ncell = batch["cycles"].shape[0]
        for c in range(ncell):
            sref = batch["summary"][c, 0]
            smg = d[sref]
            def summ(ch):
                if ch not in smg:
                    return np.zeros(0)
                v = np.asarray(smg[ch][()], float).ravel()
                return v
            S = {"Qd": summ("QDischarge"), "IR": summ("IR"),
                 "Tavg": summ("Tavg"), "Tmin": summ("Tmin")}
            if S["Qd"].size < 16 or nanmax(S["Qd"]) <= 0:
                continue
            feats_scale = [  # (lo, hi) per channel: Qd, IR, Tavg, Tmin
                (0.0, nanmax(S["Qd"])), (0.0, nanmax(S["IR"]) or 1.0),
                (10.0, 60.0), (0.0, 60.0)]
            cid = f"MATR_{os.path.basename(p)}_{c}"
            cyc += _win_cycles(S, feats_scale, cid)
    print(f"    MATR: {len(cyc)} window samples")
    return cyc


# ------------------------------------------------------------------------- CALCE
def calce_cycles():
    """Parse CALCE CS2/CX2 xlsx.  Each cell's Channel sheet: populate per-cycle
    summary (Discharge_Capacity, Internal_Resistance) then build sliding windows
    via _win_cycles.  CALCE has no reference temperature -> Tavg/Tmin set via
    an internal-resistance proxy for channel consistency."""
    import openpyxl
    cyc = []
    zips = sorted(glob.glob(os.path.join(RAW, "CALCE", "*.zip")))
    for zf in zips:
        ds = os.path.basename(zf)[:-4]
        print("  CALCE", ds)
        try:
            z = zipfile.ZipFile(zf)
        except Exception as e:
            print("    skip zip", type(e).__name__, e)
            continue
        for n in z.namelist():
            if not n.lower().endswith(".xlsx"):
                continue
            try:
                wb = openpyxl.load_workbook(io.BytesIO(z.read(n)), read_only=True, data_only=True)
            except Exception:
                continue
            # find a Channel Worksheet (holds the per-data-point table)
            sh_found = None
            for ws in wb.worksheets:
                if not ws.title.lower().startswith("channel"):
                    continue
                # Robust filtering: real chart plots are separate objects; skip any
                # sheet type whose class name mentions Chart/ChartSheet.
                cls = type(ws).__name__.lower()
                if "chart" in cls or "pivot" in cls:
                    continue
                sh_found = ws
                break
            if sh_found is None:
                continue
            rows = sh_found.iter_rows(values_only=True)
            # CALCE Channel sheets put column names in the first data-poi row;
            # find a row containing a 'Cycle' header token.
            hdr = None
            for r in rows:
                if not r:
                    continue
                cells = [str(c).strip().lower() if c is not None else "" for c in r]
                if any("cycle" in h for h in cells) and any(h for h in cells):
                    hdr = cells
                    break
            if hdr is None:
                continue
            def col(*names):
                for i, h in enumerate(hdr):
                    if any(k in h for k in names):
                        return i
                return None
            ic = col("cycle_index", "cycle")
            qdc = col("discharge_capacity")
            irc = col("internal_resistance", "internal_res", "internal")
            if ic is None or qdc is None:
                continue
            # aggregate per-cycle summary from raw per-data-point rows:
            # discharge capacity is cumulative within a cycle (last value == cycle cap)
            percyc = {}
            for r in rows:
                try:
                    cyc_i = int(float(r[ic]))
                    qd = float(r[qdc]) if qdc < len(r) and r[qdc] is not None else 0.0
                    ir = float(r[irc]) if irc is not None and irc < len(r) and r[irc] is not None else 0.0
                except (TypeError, ValueError, IndexError):
                    continue
                cur = percyc.get(cyc_i)
                if cur is None:
                    cur = [qd, ir, 1]
                    percyc[cyc_i] = cur
                else:
                    if qd > 0: cur[0] = max(cur[0], qd)   # see full-cycle discharge max
                    if ir > 0: cur[1] += ir; cur[2] += 1  # avg IR over data points
            if not percyc:
                continue
            idxs = sorted(percyc)
            Qd = np.array([percyc[k][0] for k in idxs], float)
            IR = np.array([percyc[k][1] / max(percyc[k][2], 1) for k in idxs], float)
            S = {"Qd": Qd, "IR": IR, "Tavg": np.full(len(Qd), 25.0), "Tmin": np.zeros(len(Qd))}
            if len(Qd) < 16 or nanmax(Qd) <= 0:
                continue
            feats_scale = [(0.0, nanmax(Qd)), (0.0, nanmax(IR) or 1.0), (10.0, 60.0), (0.0, 60.0)]
            cid = f"CALCE_{os.path.basename(n.rsplit('/', 1)[-1]).replace('.xlsx', '')}"
            cyc += _win_cycles(S, feats_scale, cid)
    print(f"    CALCE: {len(cyc)} window samples")
    return cyc


# ------------------------------------------------------------------- post-proc
def _downsample(a, m=MAX_TS):
    a = _f32(a)
    if a.size <= m:
        return a
    idx = np.linspace(0, a.size - 1, m).astype(int)
    return a[idx]


def normalize(cyc, ds):
    """Passthrough: sources already emit model-ready samples (SOH/RUL/_cid set
    at extraction).  Only downsample the variable-length feature channels."""
    out = []
    for c in cyc:
        out.append(dict(
            voltage=_downsample(c["voltage"]), current=_downsample(c["current"]),
            temperature=_downsample(c["temperature"]), resistance=_downsample(c["resistance"]),
            SOH=float(c["SOH"]), RUL=int(c["RUL"]), C_rate=float(c["C_rate"]),
            temp=float(c["temp"]), dSOH=float(c["dSOH"]),
            _cid=str(c.get("_cid", ds))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="ALL", choices=["NASA", "MATR", "CALCE", "ALL"])
    ap.add_argument("--max-ts", type=int, default=256)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    global MAX_TS
    MAX_TS = a.max_ts

    plan = [d for d in ("NASA", "MATR", "CALCE") if d == a.ds or a.ds == "ALL"]
    for ds in plan:
        print(f"[{ds}] extracting ...")
        if ds == "NASA":
            raw = nasa_cycles()
        elif ds == "MATR":
            raw = matr_cycles()
        else:
            raw = calce_cycles()
        print(f"[{ds}] raw cycles: {len(raw)}")
        out = normalize(raw, ds)
        save = os.path.join(OUT, f"{ds.lower()}_cycles.json")
        ser = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
               for c in out]
        with open(save, "w") as f:
            json.dump(ser, f)
        print(f"[{ds}] saved {len(ser)} cycles -> {save}")


if __name__ == "__main__":
    main()