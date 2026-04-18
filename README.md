**UNET Deep Learning Architecture**

## Overview

This repository implements a **deep learning-based surrogate model** that predicts the unsteady velocity and vorticity fields of viscous flow past a circular cylinder (von Kármán vortex street, Re ≈ 160) — without solving the governing Navier–Stokes equations at inference time.

A **U-Net convolutional encoder-decoder** learns a one-step temporal mapping:

```
[u, v](t)  →  [u, v](t+1)
```

trained on the ETH Zürich 2-D unsteady cylinder flow dataset.

---

## Repository Structure

```
unet_cylinder_wake/
├── config.py          # All hyperparameters, paths, and physical constants
├── data_loader.py     # VTI loading, preprocessing, sequence dataset creation
├── model.py           # U-Net architecture (encoder → bottleneck → decoder)
├── train.py           # Training loop, loss curve, model checkpointing
├── evaluate.py        # MSE, SSIM, R², vorticity, recirculation length metrics
├── visualize.py       # All plotting functions (7 figure types)
├── main.py            # End-to-end pipeline entry point
├── requirements.txt   # Python dependencies
└── UNET_2D_Cylinder_plots/   # (auto-created) output figures
```

---

## Dataset

Download `cylinder2d.vti` from the **ETH Zürich Computer Graphics Laboratory**:

- URL: [https://cgl.ethz.ch/research/fluid_simulation/data.php](https://cgl.ethz.ch/research/visualization/data.php)
- File: `cylinder2d.vti`
- Place it in the **root of this repository** (same directory as `main.py`).

Dataset properties:

| Property | Value |
|---|---|
| Reynolds number Re | ≈ 160 |
| Grid dimensions | 640 × 80 × 1501 (Nx × Ny × T) |
| Variables | u-velocity, v-velocity |
| Flow regime | von Kármán vortex street (periodic shedding) |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/unet-cylinder-wake.git
cd unet-cylinder-wake
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU note:** For significantly faster training, install the GPU build of TensorFlow:
> ```bash
> pip install tensorflow[and-cuda]   # TF ≥ 2.13 with CUDA 12
> ```

---

## Running the Code

### Step 1 — Full pipeline (train + evaluate + plot)

```bash
python main.py
```

This single command:
1. Loads and preprocesses `cylinder2d.vti`
2. Trains the U-Net for 20 epochs (configurable)
3. Evaluates on the held-out test set (MSE, SSIM, R², Lr)
4. Saves all figures to `UNET_2D_Cylinder_plots/`

---

###  — Train only

```bash
python train.py
python train.py --epochs 40       # custom epoch count
python train.py --resume          # continue from saved checkpoint
```

Outputs:
- `unet_cylinder.keras` — saved model weights
- `UNET_2D_Cylinder_plots/Loss_vs_Epoch_LogScale.png`
- `UNET_2D_Cylinder_plots/X_test.npy`, `y_test.npy`, `flow_stats.npy`

---

### Option C — Evaluate only (after training)

```bash
python evaluate.py
```

Requires `unet_cylinder.keras` and the `.npy` arrays saved by `train.py`.

Prints a report like:

```
───────────────────────────────────────────────────
  VELOCITY FIELD METRICS
───────────────────────────────────────────────────
  MSE                             : 0.000312
  Mean SSIM                       : 0.9874
  Overall R² Accuracy (%)         : 97.82
  Pixel Accuracy |e|<0.05 (%)     : 99.14
  U-Velocity R² Accuracy (%)      : 98.11
  V-Velocity R² Accuracy (%)      : 97.53
  ...
```

---

### Option D — Regenerate plots only

```bash
python visualize.py
# or equivalently:
python main.py --plot-only
```

Requires `y_test.npy`, `y_pred.npy`, and `flow_stats.npy` in the output directory.

---

### Option E — Skip training, use saved model

```bash
python main.py --skip-train
```

Loads `unet_cylinder.keras`, runs evaluation and generates all plots.

---

## Output Figures

| File | Description |
|---|---|
| `Loss_vs_Epoch_LogScale.png` | Training / validation loss curve (log scale) |
| `u_velocity_field.png` | Raw dataset u-velocity at t=900 |
| `v_velocity_field.png` | Raw dataset v-velocity at t=900 |
| `True_U_vs_Predicted_U.png` | Side-by-side true / predicted u |
| `True_V_vs_Predicted_V.png` | Side-by-side true / predicted v |
| `Error_Actual_U_vs_Predicted_U.png` | Pixel-wise absolute error (u) |
| `Error_Actual_V_vs_Predicted_V.png` | Pixel-wise absolute error (v) |
| `Temporal_Evolution.png` | u-velocity time series at (x=50, y=40) |
| `True_Vs_Predicted_Vorticity.png` | 2-panel vorticity comparison |
| `Vorticity_Error_Comparison_with_Accuracy.png` | 4-panel vorticity + metrics |
| `Recirculation_True_over_Predicted.png` | Centreline profile + Lr analysis |
| `Temporal_Probes_Recirculation_Threshold.png` | 3 probe locations time series |

---

## UNET Model Architecture

```
Input (H, W, 2)
│
├── Encoder Stage 1 : Conv2D(64) × 2 + MaxPool  → skip c1
├── Encoder Stage 2 : Conv2D(128) × 2 + MaxPool → skip c2
├── Encoder Stage 3 : Conv2D(256) × 2 + MaxPool → skip c3
│
├── Bottleneck       : Conv2D(512)
│
├── Decoder Stage 3 : UpSampling + concat(c3) + Conv2D(256)
├── Decoder Stage 2 : UpSampling + concat(c2) + Conv2D(128)
├── Decoder Stage 1 : UpSampling + concat(c1) + Conv2D(64)
│
Output (H, W, 2)   : Conv2D(2, kernel=1, activation=linear)
```

Optimiser: Adam · Loss: MSE · Epochs: 20 · Batch size: 4

---

## Configuration

All tunable parameters live in `config.py`. Key settings:

| Parameter | Default | Description |
|---|---|---|
| `DATA_PATH` | `cylinder2d.vti` | Path to input dataset |
| `DOWNSAMPLE_X` | 2 | Spatial downsampling in x |
| `EPOCHS` | 20 | Training epochs |
| `BATCH_SIZE` | 4 | Mini-batch size |
| `TRAIN_SPLIT` | 0.8 | Train/test chronological split |
| `SAMPLE_IDX` | 90 | Frame index for qualitative plots |
| `PROBE_DISTANCES_D` | [0.7, 1.3, 5.0] | Wake probe locations (×D) |
| `LR_TARGET` | 1.05 | Literature recirculation length (×D) |

---

## Citation

If you use this code, please cite:

```
Chauhan K., Singh S. (2026). Deep Learning–Based Surrogate Modeling for Rapid
Prediction and Visualization of Unsteady Cylinder Wake Flow Using a U-Net Architecture.
Open-SLU GSA Symposium, St. Louis, MO.
```

---

## License

MIT License — see `LICENSE` for details.
