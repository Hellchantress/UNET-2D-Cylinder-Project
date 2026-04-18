"""
evaluate.py
===========
Load a trained U-Net model, run inference on the test set, and report:
  • MSE and mean SSIM for velocity fields
  • R² accuracy and pixel accuracy for u and v channels
  • Vorticity MSE, MAE, and R²
  • Recirculation length (Lr) compared with literature (1.05 D)

Usage
-----
    python evaluate.py
"""

import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf

from config import (
    MODEL_PATH, OUTPUT_DIR,
    PIXEL_TOL, SAMPLE_IDX,
    D, X_MIN_PHYS, Y_MIN_PHYS, DOMAIN_LX, DOMAIN_LY,
    PROBE_DISTANCES_D, LR_TARGET,
)


# ── Metric helpers ────────────────────────────────────────────────────────────

def r2_score_np(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """Coefficient of determination (NumPy implementation)."""
    ss_res = np.sum((y_true - y_hat) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def vorticity(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute 2-D vorticity ω = ∂v/∂x − ∂u/∂y using central finite differences.

    Parameters
    ----------
    u, v : np.ndarray  shape (Nx, Ny)  — single time-step velocity components

    Returns
    -------
    omega : np.ndarray  shape (Nx, Ny)
    """
    dvdx = np.gradient(v, axis=0)
    dudy = np.gradient(u, axis=1)
    return dvdx - dudy


def compute_velocity_metrics(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute MSE, SSIM, R², and pixel accuracy for velocity predictions.

    Parameters
    ----------
    y_test, y_pred : np.ndarray  shape (N, Nx, Ny, 2)

    Returns
    -------
    dict of metrics
    """
    mse = float(np.mean((y_test - y_pred) ** 2))

    ssim_vals = [
        ssim(y_test[i, :, :, 0], y_pred[i, :, :, 0], data_range=1.0)
        for i in range(len(y_test))
    ]
    mean_ssim = float(np.mean(ssim_vals))

    r2_overall     = r2_score_np(y_test,        y_pred)
    r2_u           = r2_score_np(y_test[..., 0], y_pred[..., 0])
    r2_v           = r2_score_np(y_test[..., 1], y_pred[..., 1])
    pixel_acc      = float(np.mean(np.abs(y_test - y_pred) < PIXEL_TOL) * 100)

    return {
        "mse":             mse,
        "mean_ssim":       mean_ssim,
        "r2_overall_pct":  max(0.0, r2_overall) * 100,
        "pixel_acc_pct":   pixel_acc,
        "r2_u_pct":        max(0.0, r2_u) * 100,
        "r2_v_pct":        max(0.0, r2_v) * 100,
    }


def compute_vorticity_metrics(y_test: np.ndarray, y_pred: np.ndarray,
                               idx: int = SAMPLE_IDX) -> dict:
    """
    Vorticity comparison at a single sample index.

    Returns
    -------
    dict with omega arrays and scalar metrics
    """
    omega_true = vorticity(y_test[idx, :, :, 0], y_test[idx, :, :, 1])
    omega_pred = vorticity(y_pred[idx, :, :, 0], y_pred[idx, :, :, 1])
    omega_err  = np.abs(omega_true - omega_pred)

    omega_mse  = float(np.mean((omega_true - omega_pred) ** 2))
    omega_mae  = float(np.mean(omega_err))
    omega_r2   = r2_score_np(omega_true, omega_pred)

    return {
        "omega_true":     omega_true,
        "omega_pred":     omega_pred,
        "omega_err":      omega_err,
        "omega_mse":      omega_mse,
        "omega_mae":      omega_mae,
        "omega_r2_pct":   max(0.0, omega_r2) * 100,
    }


def compute_wake_geometry(y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute recirculation length (Lr) and probe x-indices.

    Returns
    -------
    dict with physical grid arrays, centerline profiles, probe indices, Lr values
    """
    NX, NY = y_test.shape[1], y_test.shape[2]
    dx = DOMAIN_LX / NX
    dy = DOMAIN_LY / NY

    x_phys = X_MIN_PHYS + (np.arange(NX) + 0.5) * dx
    y_phys = Y_MIN_PHYS + (np.arange(NY) + 0.5) * dy

    # Two centerline rows nearest to y = 0
    y_sorted = np.argsort(np.abs(y_phys - 0.0))
    y_lower, y_upper = sorted(y_sorted[:2])

    # Time-averaged centerline u
    u_mean_true = 0.5 * (y_test[:, :, y_lower, 0].mean(0) +
                          y_test[:, :, y_upper, 0].mean(0))
    u_mean_pred = 0.5 * (y_pred[:, :, y_lower, 0].mean(0) +
                          y_pred[:, :, y_upper, 0].mean(0))

    x_from_cyl_D = x_phys / D

    # Probe x-indices (clamped to valid range)
    probe_x_phys = [d * D for d in PROBE_DISTANCES_D]
    probe_x_idx  = [
        max(0, min(int(np.argmin(np.abs(x_phys - xp))), NX - 1))
        for xp in probe_x_phys
    ]

    Lr_true = _find_lr(u_mean_true, x_from_cyl_D)
    Lr_pred = _find_lr(u_mean_pred, x_from_cyl_D)

    return {
        "NX": NX, "NY": NY,
        "x_phys": x_phys,       "y_phys": y_phys,
        "x_from_cyl_D": x_from_cyl_D,
        "y_lower": y_lower,     "y_upper": y_upper,
        "u_mean_true": u_mean_true,
        "u_mean_pred": u_mean_pred,
        "probe_x_idx": probe_x_idx,
        "Lr_true": Lr_true,
        "Lr_pred": Lr_pred,
    }


def _find_lr(u_mean_1d: np.ndarray, x_from_cyl_D: np.ndarray) -> float:
    """
    Find the downstream x/D where the time-averaged centreline u-velocity
    first crosses zero from negative to positive (reattachment point).
    Returns NaN if no crossing exists.
    """
    mask = x_from_cyl_D > 0
    x_ds = x_from_cyl_D[mask]
    u_ds = u_mean_1d[mask]

    if np.min(u_ds) > 0:
        print("[evaluate] No recirculation detected: downstream u stays positive.")
        return float("nan")

    for i in range(len(u_ds) - 1):
        if u_ds[i] <= 0 < u_ds[i + 1]:
            x0, x1 = x_ds[i], x_ds[i + 1]
            u0, u1 = u_ds[i], u_ds[i + 1]
            return float(x0 - u0 * (x1 - x0) / (u1 - u0))

    print("[evaluate] Reverse flow found but no reattachment within domain.")
    return float("nan")


def print_report(vel_metrics: dict, vort_metrics: dict, wake: dict):
    """Pretty-print a summary of all computed metrics."""
    sep = "─" * 55
    print(f"\n{sep}")
    print("  VELOCITY FIELD METRICS")
    print(sep)
    print(f"  MSE                             : {vel_metrics['mse']:.6f}")
    print(f"  Mean SSIM                       : {vel_metrics['mean_ssim']:.4f}")
    print(f"  Overall R² Accuracy (%)         : {vel_metrics['r2_overall_pct']:.2f}")
    print(f"  Pixel Accuracy |e|<{PIXEL_TOL} (%) : {vel_metrics['pixel_acc_pct']:.2f}")
    print(f"  U-Velocity R² Accuracy (%)      : {vel_metrics['r2_u_pct']:.2f}")
    print(f"  V-Velocity R² Accuracy (%)      : {vel_metrics['r2_v_pct']:.2f}")

    print(f"\n{sep}")
    print("  VORTICITY METRICS")
    print(sep)
    print(f"  Omega MSE                       : {vort_metrics['omega_mse']:.6f}")
    print(f"  Omega MAE                       : {vort_metrics['omega_mae']:.6f}")
    print(f"  Omega R² Accuracy (%)           : {vort_metrics['omega_r2_pct']:.2f}")

    print(f"\n{sep}")
    print("  WAKE / RECIRCULATION LENGTH")
    print(sep)
    Lr_t = wake["Lr_true"]
    Lr_p = wake["Lr_pred"]
    print(f"  Literature target Lr            : {LR_TARGET:.2f} D")
    if not np.isnan(Lr_t):
        err = abs(Lr_t - LR_TARGET)
        print(f"  True data Lr                    : {Lr_t:.4f} D  "
              f"(error = {err:.4f} D, {100*err/LR_TARGET:.2f}%)")
    if not np.isnan(Lr_p):
        err = abs(Lr_p - LR_TARGET)
        print(f"  U-Net predicted Lr              : {Lr_p:.4f} D  "
              f"(error = {err:.4f} D, {100*err/LR_TARGET:.2f}%)")
    print(sep)


def evaluate():
    """Main entry point: load model + test data → run all metrics."""
    # Load test arrays (saved by train.py)
    X_test = np.load(os.path.join(OUTPUT_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[evaluate] Loaded model from {MODEL_PATH}")

    # Inference
    y_pred = model.predict(X_test, batch_size=BATCH_SIZE)
    np.save(os.path.join(OUTPUT_DIR, "y_pred.npy"), y_pred)

    # Metrics
    vel_metrics  = compute_velocity_metrics(y_test, y_pred)
    vort_metrics = compute_vorticity_metrics(y_test, y_pred)
    wake         = compute_wake_geometry(y_test, y_pred)

    print_report(vel_metrics, vort_metrics, wake)
    return y_test, y_pred, vel_metrics, vort_metrics, wake


if __name__ == "__main__":
    from config import BATCH_SIZE
    evaluate()
