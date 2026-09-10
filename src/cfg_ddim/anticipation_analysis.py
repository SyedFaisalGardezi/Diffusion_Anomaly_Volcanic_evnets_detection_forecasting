"""Per-eruption anticipation & detection from ALREADY-SAVED scores — no denoising,
no GPU, runs in seconds. Tests the "all 5 Whakaari" claim honestly:
  - forecasting (strict): score crosses a low-FPR threshold in the 48 h before onset
  - detection (lenient): score exceeds threshold anywhere in a 30-day pre-window
Reads cfg/results/mse_analysis/forecasting_<vol>/scores_48h.csv (written by the
eval notebook) and saves FINAL_anticipation_vs_fpr.pdf.

Run:  python3 anticipation_analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ---- paths (match the notebook) --------------------------------------------
RUN_LOCAL = False
BASE_DIR = ("."
            if RUN_LOCAL else
            "./data_external/my_code_DDPM")
MSE_DIR = os.path.join(BASE_DIR, "cfg/results/mse_analysis")
FIG_DIR = os.path.join(BASE_DIR, "cfg/results/figures")
os.makedirs(FIG_DIR, exist_ok=True)

SAMPLES_PER_H = 6                     # 10-min cadence
EXCL_DAYS = 30                        # Ardid negative-class exclusion (SI Fig S1)
OP_FPR = 0.10                         # fixed operating point for the single-number anticipation count
BEST_WIN = {"whakaari": 48, "ruapehu": 48, "pavlof": 48}   # FIXED 48 h (Ardid 2-day window), all volcanoes
# Score source. "mse" = the summed-residual (scores_48h.csv). "dsar_weighted" uses the
# per-channel file scores_perchannel.csv (run rescore_perchannel.py first on the GPU) ->
# 2*DSAR + sum(all channels); DSAR is the phreatic precursor (Ardid 2022). Same 48 h
# median / +-30 d / 10% FPR protocol -- only the score changes, never the threshold.
SCORE = "mse"                         # "mse" | "dsar_weighted"
ERUPTIONS = {
    "whakaari": ["2012-08-04 16:52", "2013-08-19 22:23", "2013-10-03 12:35", "2016-04-27 09:37", "2019-12-09 01:11"],
    "ruapehu":  ["2006-10-04", "2007-09-25"],
    "pavlof":   ["2014-05-31", "2014-11-13", "2016-03-28"],
}

def _causal(x, k): return pd.Series(x).rolling(int(max(1, k)), min_periods=1).median().to_numpy()  # Ardid 2-day MEDIAN
def _auc(s, y):
    y = np.asarray(y).astype(int)
    return np.nan if (y.sum() == 0 or y.sum() == len(y)) else roc_auc_score(y, np.asarray(s, float))

def _raw_score(vol):
    """Return (time, raw_per_sample_score, label) for the selected SCORE source."""
    d = os.path.join(MSE_DIR, f"forecasting_{vol}")
    if SCORE == "dsar_weighted":
        p = os.path.join(d, "scores_perchannel.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(p + "  (run rescore_perchannel.py on the GPU first)")
        df = pd.read_csv(p, parse_dates=["time"])
        chans = [c for c in df.columns if c.startswith("res_")]
        allsum = df[chans].to_numpy(float).sum(axis=1)
        score = 2.0 * df["res_dsar"].to_numpy(float) + allsum
    else:
        p = os.path.join(d, "scores_48h.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        df = pd.read_csv(p, parse_dates=["time"])
        score = df["mse"].to_numpy(float)
    return df["time"].to_numpy().astype("datetime64[ns]"), score, df["label"].to_numpy().astype(int)

def load(vol):
    t, raw, y = _raw_score(vol)
    return t, _causal(raw, BEST_WIN[vol] * SAMPLES_PER_H), y

def ardid_keep(t, vol):
    excl = np.zeros(len(t), bool)
    for e in ERUPTIONS[vol]:
        te = np.datetime64(pd.Timestamp(e))
        excl |= (t >= te - np.timedelta64(EXCL_DAYS, "D")) & (t <= te + np.timedelta64(EXCL_DAYS, "D"))
    return excl

def anticipation_table(vol, fprs=(0.01, 0.02, 0.05, 0.10, 0.20)):
    t, s, y = load(vol)
    keep = (y == 1) | (~ardid_keep(t, vol))
    neg = s[keep & (y == 0)]
    evs = [np.datetime64(pd.Timestamp(e)) for e in ERUPTIONS[vol]]
    rows = []
    for fpr in fprs:
        thr = np.quantile(neg, 1 - fpr); h48 = h30 = 0; leads = []
        for te in evs:
            w48 = (t >= te - np.timedelta64(48, "h")) & (t < te)
            w30 = (t >= te - np.timedelta64(30, "D")) & (t < te)
            if w48.any() and (s[w48] > thr).any():
                h48 += 1; leads.append((te - t[w48 & (s > thr)][0]) / np.timedelta64(1, "h"))
            if w30.any() and (s[w30] > thr).any(): h30 += 1
        n = len(evs)
        rows.append([f"{int(fpr*100)}%", f"{h48}/{n}", f"{h30}/{n}", f"{np.mean(leads):.1f}" if leads else "-"])
    return pd.DataFrame(rows, columns=["FPR", "anticip(48h)", "detect(30d)", "lead h"])

def per_event(vol, fpr=0.05):
    t, s, y = load(vol)
    keep = (y == 1) | (~ardid_keep(t, vol))
    thr = np.quantile(s[keep & (y == 0)], 1 - fpr)
    rows = []
    for e in ERUPTIONS[vol]:
        te = np.datetime64(pd.Timestamp(e))
        w48 = (t >= te - np.timedelta64(48, "h")) & (t < te)
        w30 = (t >= te - np.timedelta64(30, "D")) & (t < te)
        rows.append([e, "Y" if (w48.any() and (s[w48] > thr).any()) else "n",
                     "Y" if (w30.any() and (s[w30] > thr).any()) else "n",
                     f"{s[w48].max():.3f}" if w48.any() else "-", f"{thr:.3f}"])
    return pd.DataFrame(rows, columns=["eruption", "48h", "30d", "peak(48h)", f"thr@{int(fpr*100)}%"])

def main():
    print(f"SCORE = {SCORE!r} | fixed {BEST_WIN['whakaari']} h causal median | +/-{EXCL_DAYS} d negatives | OP {int(OP_FPR*100)}% FPR")
    plt.figure(figsize=(6, 4))
    for vol, c in [("whakaari", "tab:blue"), ("ruapehu", "tab:green"), ("pavlof", "tab:red")]:
        try:
            print(f"\n=== {vol} (best window {BEST_WIN[vol]} h) ===")
            t, s, y = load(vol)
            keep = (y == 1) | (~ardid_keep(t, vol))
            print(f"  Ardid-exact AUC = {_auc(s[keep], y[keep]):.3f}")
            print(anticipation_table(vol).to_string(index=False))
            print(per_event(vol).to_string(index=False))
        except FileNotFoundError as ex:
            print(f"  skip {vol}: {ex}"); continue
        # anticipation-vs-FPR curve
        neg = s[keep & (y == 0)]; evs = [np.datetime64(pd.Timestamp(e)) for e in ERUPTIONS[vol]]
        fr = np.linspace(0.005, 0.3, 40); frac = []
        for fpr in fr:
            thr = np.quantile(neg, 1 - fpr); h = 0
            for te in evs:
                w = (t >= te - np.timedelta64(48, "h")) & (t < te)
                if w.any() and (s[w] > thr).any(): h += 1
            frac.append(h / len(evs))
        plt.plot(fr * 100, frac, "o-", ms=3, color=c, label=vol.capitalize())

    # ---- single-number anticipation at the fixed operating point (table-ready) ----
    ARDID = {"whakaari": "4/5", "ruapehu": "1/2", "pavlof": "0/2"}
    print(f"\n=== Anticipation @ fixed {int(OP_FPR*100)}% FPR (compare to Ardid) ===")
    srows = []
    for vol in ["whakaari", "ruapehu", "pavlof"]:
        if not os.path.exists(os.path.join(MSE_DIR, f"forecasting_{vol}", "scores_48h.csv")):
            continue
        t, s, y = load(vol)
        keep = (y == 1) | (~ardid_keep(t, vol)); neg = s[keep & (y == 0)]
        thr = np.quantile(neg, 1 - OP_FPR)
        evs = [np.datetime64(pd.Timestamp(e)) for e in ERUPTIONS[vol]]
        h, leads = 0, []
        for te in evs:
            w = (t >= te - np.timedelta64(48, "h")) & (t < te)
            if w.any() and (s[w] > thr).any():
                h += 1; leads.append((te - t[w & (s > thr)][0]) / np.timedelta64(1, "h"))
        srows.append([vol.capitalize(), f"{h}/{len(evs)}", ARDID[vol],
                      f"{np.mean(leads):.1f}" if leads else "-"])
    sdf = pd.DataFrame(srows, columns=["Volcano", "DDPM anticipated", "Ardid", "mean lead h"])
    print(sdf.to_string(index=False))
    sdf.to_csv(os.path.join(FIG_DIR, "FINAL_anticipation_counts.csv"), index=False)

    # ---- score-variant check (free; same 48 h-median protocol & 10% FPR) ----
    # Tests whether a different saved anomaly score (cosine / combined) legitimately
    # anticipates more than the MSE residual, WITHOUT loosening the threshold.
    def _antic(s, t, y, vol):
        keep = (y == 1) | (~ardid_keep(t, vol))
        thr = np.quantile(s[keep & (y == 0)], 1 - OP_FPR)
        h = 0
        for e in ERUPTIONS[vol]:
            te = np.datetime64(pd.Timestamp(e)); w = (t >= te - np.timedelta64(48, "h")) & (t < te)
            if w.any() and (s[w] > thr).any(): h += 1
        return _auc(s[keep], y[keep]), h, len(ERUPTIONS[vol])
    print(f"\n=== Score-variant check (48 h median, AUC | antic@{int(OP_FPR*100)}%FPR) ===")
    vrows = []
    for vol in ["whakaari", "ruapehu", "pavlof"]:
        p = os.path.join(MSE_DIR, f"forecasting_{vol}", "scores_48h.csv")
        if not os.path.exists(p): continue
        df = pd.read_csv(p, parse_dates=["time"]); t = df["time"].to_numpy().astype("datetime64[ns]")
        y = df["label"].to_numpy().astype(int)
        row = [vol.capitalize()]
        for col in ["mse", "cosine", "combined"]:
            if col not in df.columns: row.append("-"); continue
            a, h, n = _antic(_causal(df[col].to_numpy(float), 48 * SAMPLES_PER_H), t, y, vol)
            row.append(f"{a:.3f}|{h}/{n}")
        vrows.append(row)
    print(pd.DataFrame(vrows, columns=["Volcano", "mse", "cosine", "combined"]).to_string(index=False))

    plt.axvline(OP_FPR * 100, ls=":", color="grey", lw=1, label=f"operating point {int(OP_FPR*100)}% FPR")
    plt.xlabel("FPR (%)"); plt.ylabel("fraction anticipated (48 h)")
    plt.title("Anticipation vs FPR"); plt.legend(); plt.tight_layout()
    out = os.path.join(FIG_DIR, "FINAL_anticipation_vs_fpr.pdf")
    plt.savefig(out, bbox_inches="tight"); print("\nsaved ->", out)

if __name__ == "__main__":
    main()
