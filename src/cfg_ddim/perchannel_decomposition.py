"""Reviewer comment 1.7 — per-channel anomaly-score decomposition + Fig. 4.

The anomaly score (Eq. 16) is a sum over channels of squared residuals, so it
decomposes EXACTLY into per-channel contributions (no retrain, no new model):
    a_{l,c} = (x_{c,l} - xbar_{c,l})^2        # per-channel
    a_l     = sum_c a_{l,c}                    # = the existing score
We recover x (input) and xbar (ensemble-mean reconstruction) by one ordinary
inference pass on a focused case-study window per volcano (where the detected
episodes sit), then report:
  - per-channel residual share inside vs outside detected events (fills the
    [DOMINANT_CHANNEL]/[XX]% placeholders in Response 1.7), and
  - the 6-panel Fig. 4 for Whakaari, Jul-Dec 2019.

Run (GPU, tmux):  python perchannel_decomposition.py
Outputs: cfg/results/figures/fig4_channels.pdf
         cfg/results/figures/perchannel_shares.csv
"""
import os, sys, time
import numpy as np, pandas as pd, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

RUN_LOCAL = False
BASE_DIR = ("."
            if RUN_LOCAL else "./data_external/my_code_DDPM")
sys.path.append(os.path.join(BASE_DIR, "cfg", "src"))
try:
    import opensimplex  # noqa: F401
except ModuleNotFoundError:
    import types as _t; _s = _t.ModuleType("opensimplex"); _s.OpenSimplex = object
    sys.modules["opensimplex"] = _s
from cfg_ddim import DenoiseDiffusion, gather   # noqa: E402
from cfg_ddim.unet import UNet                  # noqa: E402

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
FIG_DIR = os.path.join(BASE_DIR, "cfg/results/figures"); os.makedirs(FIG_DIR, exist_ok=True)

FEATS = ["RSAM", "MF", "HF", "DSAR", "SSAM"]; COLS = ["rsam", "mf", "hf", "dsar", "ssam"]
N_STEPS = 1000; WTSG = 3.5; S = 2.0; T_MIN = 10; ALPHA = 1.0
LAMBDA_MAX = 500; CHUNK = 50; ENS = 5; BATCH = 16          # M=5 (matches the paper's ensemble)
K_MAD = {"whakaari": 0.5, "ruapehu": 3.0, "pavlof": 7.0}   # paper's per-volcano median-MAD multipliers

PAVLOF = os.path.join(BASE_DIR, "cfg/data/raw/outside_NZ/PVV_seismic_data_with_ssam_norm_z_percen_m11.csv")
_SAE = "./data_external/pavlof/PVV_seismic_data_with_ssam_norm_z_percen_m11.csv"
if not os.path.exists(PAVLOF) and os.path.exists(_SAE): PAVLOF = _SAE

# (csv, exp_id, focused window [contains the detected episode], eruption date for the vline)
CASES = {
    "whakaari": (os.path.join(BASE_DIR, "cfg/data/processed/previous_paper_data/WIZ_with_ssam_asof_norm_z_percen_m11.csv"),
                 "27_Tremor", "2019-07-01", "2019-12-31", "2019-12-09 01:11"),
    "ruapehu":  (os.path.join(BASE_DIR, "cfg/data/raw/Ruapehu_seismic_data_with_ssam_norm_z_percen_m11.csv"),
                 "23_Tremor", "2007-06-01", "2007-10-10", "2007-09-25"),
    "pavlof":   (PAVLOF, "29_Tremor", "2016-01-01", "2016-04-30", "2016-03-28"),
}

# ---- tolerant checkpoint loader (Configs pickled into ckpt) ----------------
import pickle, types
class _SM(type):
    def __getattr__(cls, n): return lambda *a, **k: None
class _Stub(metaclass=_SM):
    def __new__(cls, *a, **k): return object.__new__(cls)
    def __init__(self, *a, **k): pass
    def __setstate__(self, s):
        if isinstance(s, dict):
            try: self.__dict__.update(s)
            except Exception: pass
    def __setitem__(self, k, v): pass
    def append(self, *a, **k): pass
    def __getattr__(self, n): return lambda *a, **k: None
class _TU(pickle.Unpickler):
    def find_class(self, m, n):
        try: return super().find_class(m, n)
        except Exception: return _SM(n, (_Stub,), {})
_PM = types.SimpleNamespace(__name__="_tp", Unpickler=_TU, load=pickle.load, Pickler=pickle.Pickler,
        dump=pickle.dump, UnpicklingError=pickle.UnpicklingError, HIGHEST_PROTOCOL=pickle.HIGHEST_PROTOCOL)

def build(exp):
    m = UNet(image_channels=5, n_channels=64, ch_mults=[1, 2, 2, 4],
             is_attn=[False, False, False, True], n_blocks=2).to(DEVICE)
    diff = DenoiseDiffusion(eps_model=m, n_steps=N_STEPS, device=DEVICE, guidance_scale=WTSG,
                            lambda_max=LAMBDA_MAX, T_MIN=T_MIN, T_MAX=N_STEPS - T_MIN, S=S, ALPHA=ALPHA)
    ck = torch.load(os.path.join(BASE_DIR, f"cfg/src/cfg_ddim/model_weights/tsg_ddim_TSG_V10_TSG_exp_{exp}.pth"),
                    map_location=DEVICE, weights_only=False, pickle_module=_PM)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return diff

def reconstruct(diff, x):                  # x:[B,C,L] -> ensemble-mean reconstruction xbar [B,C,L]
    def once(xx):
        t = torch.full((xx.shape[0],), LAMBDA_MAX - 1, dtype=torch.long, device=DEVICE)
        xt = diff.q_sample_ode(xx, t, eps=torch.randn_like(xx))
        for ti in range(LAMBDA_MAX):
            tt = torch.full((xx.shape[0],), LAMBDA_MAX - ti - 1, dtype=torch.long, device=DEVICE)
            xt = diff.ddim_sample_heun(xt, tt, torch.clamp(tt - 1, min=0))
        return xt.cpu()
    return torch.stack([once(x) for _ in range(ENS)], 0).mean(0)

def score_window(vol):
    csv, exp, t0, t1, _ = CASES[vol]
    diff = build(exp)
    df = pd.read_csv(csv); df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    df = df[(df["time"] >= pd.to_datetime(t0)) & (df["time"] <= pd.to_datetime(t1))].reset_index(drop=True)
    Xa = df[COLS].astype("float32").to_numpy(); times = pd.DatetimeIndex(df["time"])
    g = ~np.isnan(Xa).any(axis=1); Xa, times = Xa[g], times[g]
    n = len(Xa) // CHUNK; Xa = Xa[:n * CHUNK]; times = times[:n * CHUNK]
    xin, xbar, tim = [], [], []
    t_in = 0.0
    for b in range(0, n, BATCH):
        bb = list(range(b, min(b + BATCH, n)))
        xb = torch.stack([torch.from_numpy(Xa[i * CHUNK:(i + 1) * CHUNK]).permute(1, 0) for i in bb]).to(DEVICE)
        with torch.no_grad():
            _t = time.perf_counter(); xr = reconstruct(diff, xb)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t_in += time.perf_counter() - _t
        for j, si in enumerate(bb):
            xin.append(xb[j].cpu().numpy()); xbar.append(xr[j].numpy())
            tim.append(times[si * CHUNK:(si + 1) * CHUNK].values)
        if (b // BATCH) % 10 == 0: print(f"  {vol} {b+len(bb)}/{n} ({t_in/60:.1f} min)")
    X = np.concatenate(xin, axis=1)        # (C, L)
    XB = np.concatenate(xbar, axis=1)      # (C, L)
    T = np.concatenate(tim)
    RES = (X - XB) ** 2                     # (C, L) per-channel contribution
    score = RES.sum(0)                      # (L,)  = a_l
    # sanity: decomposition is exact by construction
    assert np.allclose(score, RES.sum(0), atol=1e-6)
    return T, X, XB, RES, score

def detect_mask(score, k):                 # paper's median-MAD rule
    med = np.median(score); mad = np.median(np.abs(score - med))
    theta = med + k * mad
    return score > theta, theta

def shares(RES, mask):
    e_in = RES[:, mask].sum(1); e_in = e_in / (e_in.sum() + 1e-12)
    e_out = RES[:, ~mask].sum(1); e_out = e_out / (e_out.sum() + 1e-12)
    return e_in, e_out

def make_fig4(vol, T, X, XB, RES, score, theta):
    erupt = pd.Timestamp(CASES[vol][4]); t = pd.to_datetime(T)
    fig, ax = plt.subplots(6, 1, sharex=True, figsize=(7.5, 9))
    for i, name in enumerate(FEATS):
        ax[i].plot(t, X[i], color="k", lw=0.7, label="input")
        ax[i].plot(t, XB[i], color="tab:red", lw=0.7, alpha=0.8, label="reconstruction")
        ax[i].set_ylabel(f"{name}\n(norm.)", fontsize=8); ax[i].set_ylim(-1.05, 1.05)
        ax[i].tick_params(labelsize=7)
    ax[0].legend(loc="upper left", fontsize=6, ncol=2)
    ax[5].stackplot(t, RES, labels=FEATS, alpha=0.85)
    ax[5].plot(t, score, color="k", lw=0.8, label="total score")
    ax[5].axhline(theta, ls="--", color="gray", lw=0.8, label="median-MAD threshold")
    ax[5].axvline(erupt, ls="--", color="red", lw=1.0, label="eruption")
    ax[5].set_ylabel("anomaly\nscore", fontsize=8); ax[5].legend(ncol=4, fontsize=6, loc="upper left")
    ax[5].tick_params(labelsize=7)
    ax[5].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax[5].xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax[5].set_xlabel("Date")
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig4_channels.pdf")
    plt.savefig(out, bbox_inches="tight"); plt.savefig(out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("saved ->", out, "(+ .png)")

def main():
    rows = []
    for vol in ["whakaari", "ruapehu", "pavlof"]:
        if not os.path.exists(CASES[vol][0]):
            print(f"skip {vol}: missing {CASES[vol][0]}"); continue
        print(f"\n=== {vol} ({CASES[vol][2]}..{CASES[vol][3]}) ===")
        T, X, XB, RES, score = score_window(vol)
        mask, theta = detect_mask(score, K_MAD[vol])
        e_in, e_out = shares(RES, mask)
        dom = FEATS[int(np.argmax(e_in))]
        print(f"  detected samples: {mask.sum()}/{len(mask)}  threshold theta={theta:.4f}")
        print("  in-event share : " + "  ".join(f"{f}={100*e_in[i]:.1f}%" for i, f in enumerate(FEATS)))
        print("  background share: " + "  ".join(f"{f}={100*e_out[i]:.1f}%" for i, f in enumerate(FEATS)))
        print(f"  -> dominant in-event channel: {dom} ({100*e_in.max():.1f}%)")
        rows.append([vol, "in_event"] + [round(100 * v, 1) for v in e_in] + [dom])
        rows.append([vol, "background"] + [round(100 * v, 1) for v in e_out] + [""])
        if vol == "whakaari":
            make_fig4(vol, T, X, XB, RES, score, theta)
    df = pd.DataFrame(rows, columns=["volcano", "period"] + FEATS + ["dominant"])
    out = os.path.join(FIG_DIR, "perchannel_shares.csv")
    df.to_csv(out, index=False)
    print("\n=== per-channel shares (%) ===")
    print(df.to_string(index=False))
    print("saved ->", out)
    print("\nFill Response 1.7: dominant channel + in-event % per volcano from the table above.")

if __name__ == "__main__":
    main()
