# Pipeline walkthrough

How the pieces fit together, from raw features to a forecasting-AUC number.

```
feature CSVs (C=5, 10-min)
        │
        ▼
 experiment_cfg_ddim.Configs / Dataset      # segmentation into (C x S)=(5 x 50) windows
        │        (train on clean/non-eruptive segments only)
        ▼
 training_model_cfg_ddim.py  ──trains──►  unet.UNet  (1-D U-Net denoiser)
        │                                    ▲
        ▼                                    │ eps_theta(x_t, t)
 checkpoint  .pth  ──────────────────────────┘
        │
        ▼
 evaluate_cgf_ddim_tremor.py                 # per-segment reconstruction
   ├─ __init__.DenoiseDiffusion              #   partial-noise start x_{lambda_max}
   │     ├─ Time-Step Guidance (w_TSG)       #   guided score
   │     └─ Heun 2nd-order probability-flow  #   deterministic reverse ODE
   ├─ anomaly score = sum_c (x - x_hat)^2    # per-time-step, per-channel
   ├─ trimmed median–MAD threshold           # detections
   └─ ROC AUC vs eruptive_periods            # forecasting skill
        │
        ▼
 metrics.py  (F1_K-AUC, ROC_K-AUC, PA%K)      # benchmark protocol
```

## Modules

### `src/cfg_ddim/__init__.py` — `DenoiseDiffusion`
The core diffusion object. Holds the beta schedule (`linspace(1e-4, 2e-2, T)`),
`alpha`, `alpha_bar`. Implements:
- the forward `q(x_t | x_0)` corruption (Gaussian, with an optional simplex-noise
  variant via `opensimplex`);
- the reverse step with **Time-Step Guidance**: the guided noise estimate mixes the
  denoiser evaluated at the true and a perturbed time embedding, weighted by
  `guidance_scale` (= `w_TSG`);
- **Heun** second-order integration of the probability-flow ODE (predictor Euler step
  + slope-averaged corrector);
- the **partial-noise** reverse trajectory starting at `lambda_max < T`.
Key constructor args: `n_steps` (T), `guidance_scale`, `lambda_max`, `T_MIN`,
`T_MAX`, `S`, `ALPHA`.

### `src/cfg_ddim/unet.py` — `UNet`
1-D U-Net with sinusoidal time embeddings (`TimeEmbedding`), residual blocks,
attention, down/up sampling. Input `(B, C=5, S=50)`.

### `src/cfg_ddim/utils.py` — `gather`
Indexes a schedule tensor by timestep and reshapes for broadcasting against
`(B, C, S)` tensors.

### `src/cfg_ddim/experiment_cfg_ddim.py` — `Configs`, `Dataset`
`labml` `BaseConfigs` subclass defining the model, optimizer, data loader, and
training loop, plus a `Dataset` that reads a feature CSV and segments it into
`(C x S)` windows. `image_channels` must be 5 for tremor. `dataset` selects the data
branch (`Tremor`, `SWAT`, `WADI`, `Yahoo`, `Synthetic`).

### `src/cfg_ddim/training_model_cfg_ddim.py`
Training entry point. Builds `Configs`, overrides them via `experiment.configs(...)`,
`configs.init()`, registers the model, and runs. Edit the config block to point at
your **clean** training CSV and set `exp`, `chunk_size`, `lambda_max`,
`guidance_scale`, `epochs`.

### `src/cfg_ddim/evaluate_cgf_ddim_tremor.py`
Volcano evaluation. `main()` selects the volcano (checkpoint tag `exp`, `test_path`,
`w_TSG`, date window). For each segment it reconstructs with the partial-noise +
TSG + Heun reverse process, computes the per-time-step anomaly score, applies the
median–MAD threshold, and reports timing/AUC. Per-segment MSE arrays are cached to
`./cfg/results/mse_analysis/` so scoring can be re-run offline without a GPU pass.

### `src/cfg_ddim/evaluate_cgf_ddim.py`
Same reconstruction engine, wired to the multivariate-AD benchmarks (SWaT, WaDi,
synthetic). Reports `F1_K-AUC` / `ROC_K-AUC`.

### `src/cfg_ddim/testing.py`
Lighter standalone reconstruction/scoring harness — useful for sanity-checking a
checkpoint on a single file.

### `src/cfg_ddim/metrics.py`
Threshold-independent metrics: `F1_K-AUC`, `ROC_K-AUC`, and the point-adjust-@K
(`PA%K`) protocol (via `tadpak`) used for consistency with the baselines.

### `notebooks/DDPM_forecasting_evaluation_GPU.ipynb`
The forecasting-skill evaluation (Ardid-2025 protocol): 48-hour positive window,
±30-day buffer, single fixed 48-hour causal-median smoothing, full-record ROC AUC.
Regenerate with `_build_eval_gpu_nb.py`.

## Reproduction order

1. Build the 5-feature CSVs (see README → Data).
2. Train one model per volcano on its clean CSV → checkpoint under `model_weights/`.
3. Run `evaluate_cgf_ddim_tremor.py` per volcano for detections + AUC.
4. Run the forecasting notebook for the full-record AUC table.
5. Run `evaluate_cgf_ddim.py` for the benchmark table.
