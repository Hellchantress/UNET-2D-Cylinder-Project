"""
config.py
=========
Central configuration for the U-Net cylinder wake surrogate model.
Edit values here; all other modules import from this file.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = "cylinder2d.vti"          # ETH Zürich VTI dataset
OUTPUT_DIR  = "UNET_2D_Cylinder_plots"  # directory for saved figures
MODEL_PATH  = "unet_cylinder.keras"     # saved Keras model

# ── Data / Grid ───────────────────────────────────────────────────────────────
DOWNSAMPLE_X = 2          # spatial downsampling factor in x (every 2nd cell)
DOWNSAMPLE_Y = 1          # spatial downsampling factor in y (every cell)

# Physical domain parameters (simulation length units)
D           = 0.125       # cylinder diameter
X_MIN_PHYS  = -0.5        # physical domain left edge
Y_MIN_PHYS  = -0.5        # physical domain bottom edge
DOMAIN_LX   = 8.0         # stream-wise domain length
DOMAIN_LY   = 1.0         # cross-stream domain length

# ── Model ─────────────────────────────────────────────────────────────────────
TRAIN_SPLIT  = 0.8        # fraction of samples used for training
VAL_SPLIT    = 0.2        # fraction of train used for validation (Keras param)
EPOCHS       = 20
BATCH_SIZE   = 4
OPTIMIZER    = "adam"
LOSS         = "mse"

# ── Evaluation ────────────────────────────────────────────────────────────────
PIXEL_TOL    = 0.05       # |error| threshold for pixel-accuracy metric
SAMPLE_IDX   = 90         # single-frame index used in qualitative plots

# ── Wake analysis ─────────────────────────────────────────────────────────────
LR_TARGET           = 1.05                    # literature recirculation length (×D)
PROBE_DISTANCES_D   = [0.7, 1.3, 5.0]         # probe locations in diameters
TEMPORAL_PROBE      = (50, 40)                 # (x_idx, y_idx) for single-point time series

# ── Misc ──────────────────────────────────────────────────────────────────────
FIGURE_DPI_LOW  = 1200
FIGURE_DPI_HIGH = 3600

os.makedirs(OUTPUT_DIR, exist_ok=True)
