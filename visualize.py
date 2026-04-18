"""
visualize.py
============
All plotting routines for the U-Net cylinder wake surrogate model.

Functions
---------
plot_raw_velocity_fields   – actual u/v fields from the dataset
plot_velocity_comparison   – true vs predicted u and v side-by-side
plot_absolute_error        – pixel-wise absolute error maps
plot_temporal_evolution    – single-probe time series
plot_vorticity             – true vs predicted vorticity + error
plot_recirculation         – centreline profile + Lr analysis
plot_temporal_probes       – 3-probe temporal evolution with threshold

Usage
-----
    python visualize.py        # run all plots (requires pre-saved .npy outputs)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    OUTPUT_DIR,
    SAMPLE_IDX, TEMPORAL_PROBE,
    PROBE_DISTANCES_D, LR_TARGET, D,
    FIGURE_DPI_LOW, FIGURE_DPI_HIGH,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _savefig(name: str, dpi: int = FIGURE_DPI_LOW):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[visualize] Saved → {path}")
    plt.close()


# ── 1. Raw velocity fields ────────────────────────────────────────────────────

def plot_raw_velocity_fields(flow: np.ndarray, t: int = 900):
    """
    Show u and v velocity fields at a single time step.

    Parameters
    ----------
    flow : np.ndarray  shape (T, Nx, Ny, 2)  normalised
    t    : int  time index to visualise
    """
    for ch, name in enumerate(["u_velocity", "v_velocity"]):
        plt.figure()
        plt.imshow(flow[t, :, :, ch].T, origin='lower')
        plt.title(f"Actual {name.replace('_', ' ')} field (t={t})")
        plt.colorbar()
        _savefig(f"{name}_field.png", dpi=FIGURE_DPI_HIGH)


# ── 2. True vs predicted velocity ─────────────────────────────────────────────

def plot_velocity_comparison(y_test: np.ndarray, y_pred: np.ndarray,
                              idx: int = SAMPLE_IDX):
    """Side-by-side true / predicted u and v fields."""
    for ch, label in enumerate(["u", "v"]):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].imshow(y_test[idx, :, :, ch].T, origin='lower')
        axes[0].set_title(f"True {label}")
        axes[0].set_xlabel("x-index")
        axes[0].set_ylabel(label)

        axes[1].imshow(y_pred[idx, :, :, ch].T, origin='lower')
        axes[1].set_title(f"Predicted {label}")
        axes[1].set_xlabel("x-index")
        axes[1].set_ylabel(label)

        plt.tight_layout()
        _savefig(f"True_{label.upper()}_vs_Predicted_{label.upper()}.png",
                 dpi=FIGURE_DPI_HIGH)


# ── 3. Absolute error maps ────────────────────────────────────────────────────

def plot_absolute_error(y_test: np.ndarray, y_pred: np.ndarray,
                        idx: int = SAMPLE_IDX):
    """Pixel-wise |true − pred| for u and v."""
    error = np.abs(y_test[idx] - y_pred[idx])
    for ch, label in enumerate(["u", "v"]):
        plt.figure()
        plt.imshow(error[:, :, ch].T, origin='lower')
        plt.title(f"Abs. Error: {label}-velocity (frame {idx})")
        plt.xlabel("x-index")
        plt.ylabel(label)
        plt.colorbar()
        _savefig(f"Error_Actual_{label.upper()}_vs_Predicted_{label.upper()}.png",
                 dpi=FIGURE_DPI_HIGH)


# ── 4. Single-probe temporal evolution ───────────────────────────────────────

def plot_temporal_evolution(y_test: np.ndarray, y_pred: np.ndarray,
                             probe: tuple = TEMPORAL_PROBE):
    """
    Normalised u-velocity over time at a single spatial location.

    Parameters
    ----------
    probe : (x_idx, y_idx)
    """
    xi, yi = probe
    true_series = y_test[:, xi, yi, 0]
    pred_series = y_pred[:, xi, yi, 0]
    time_steps  = np.arange(len(true_series))

    plt.figure(figsize=(10, 7))
    plt.plot(time_steps, true_series, label='True')
    plt.plot(time_steps, pred_series, label='Predicted')
    plt.xlabel("Time Step / Frame Index")
    plt.ylabel("Normalised u-velocity")
    plt.title(f"Temporal Evolution of u-velocity at (x={xi}, y={yi})")
    plt.legend()
    plt.grid(True)
    _savefig("Temporal_Evolution.png", dpi=FIGURE_DPI_HIGH)


# ── 5. Vorticity panels ───────────────────────────────────────────────────────

def plot_vorticity(vort_metrics: dict, idx: int = SAMPLE_IDX):
    """
    4-panel: true vorticity | predicted | abs error | with metrics in title.

    Parameters
    ----------
    vort_metrics : dict  returned by evaluate.compute_vorticity_metrics()
    """
    omega_true = vort_metrics["omega_true"]
    omega_pred = vort_metrics["omega_pred"]
    omega_err  = vort_metrics["omega_err"]

    # 2-panel (true vs predicted)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(omega_true.T, origin='lower')
    axes[0].set_title("True Vorticity")
    axes[1].imshow(omega_pred.T, origin='lower')
    axes[1].set_title("Predicted Vorticity")
    plt.tight_layout()
    _savefig("True_Vs_Predicted_Vorticity.png", dpi=FIGURE_DPI_HIGH)

    # 4-panel with error and metrics
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    im0 = axes[0].imshow(omega_true.T, origin='lower')
    axes[0].set_title("True Vorticity")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(omega_pred.T, origin='lower')
    axes[1].set_title("Predicted Vorticity")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(omega_err.T, origin='lower')
    axes[2].set_title("Abs. Vorticity Error")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    axes[3].axis('off')
    axes[3].text(0.5, 0.5,
                 f"MSE = {vort_metrics['omega_mse']:.6f}\n"
                 f"MAE = {vort_metrics['omega_mae']:.6f}\n"
                 f"R² acc = {vort_metrics['omega_r2_pct']:.2f}%",
                 ha='center', va='center', fontsize=12,
                 transform=axes[3].transAxes)

    fig.suptitle(
        f"Vorticity Comparison (frame {idx})\n"
        f"MSE={vort_metrics['omega_mse']:.6f}  "
        f"MAE={vort_metrics['omega_mae']:.6f}  "
        f"Acc={vort_metrics['omega_r2_pct']:.2f}%",
        fontsize=11)
    plt.tight_layout()
    _savefig("Vorticity_Error_Comparison_with_Accuracy.png", dpi=FIGURE_DPI_HIGH)


# ── 6. Centreline recirculation profile ───────────────────────────────────────

def plot_recirculation(wake: dict):
    """
    True-over-predicted centreline u-velocity profile with Lr markers.

    Parameters
    ----------
    wake : dict  returned by evaluate.compute_wake_geometry()
    """
    x_from_cyl_D = wake["x_from_cyl_D"]
    u_mean_true  = wake["u_mean_true"]
    u_mean_pred  = wake["u_mean_pred"]
    x_phys       = wake["x_phys"]
    probe_x_idx  = wake["probe_x_idx"]
    Lr_true      = wake["Lr_true"]
    Lr_pred      = wake["Lr_pred"]

    fig, ax1 = plt.subplots(figsize=(9, 5))

    ax1.plot(x_from_cyl_D, u_mean_pred, color='darkorange', lw=2,
             label='U-Net Predicted')
    ax1.plot(x_from_cyl_D, u_mean_true, color='steelblue', lw=0,
             marker='^', markersize=5, markevery=1, label='True u')

    ax1.axhline(0,          color='k',    lw=0.8, ls=':')
    ax1.axvline(0,          color='gray', lw=0.8, ls='--', label='Cylinder centre')
    ax1.axvline(LR_TARGET,  color='blue', lw=1.4, ls='--',
                label=f'Target Lr = {LR_TARGET:.2f} D')

    if not np.isnan(Lr_true):
        ax1.axvline(Lr_true, color='steelblue', lw=1.4, ls='-.',
                    label=f'True Lr = {Lr_true:.3f} D')
    if not np.isnan(Lr_pred):
        ax1.axvline(Lr_pred, color='darkorange', lw=1.4, ls='-.',
                    label=f'Predicted Lr = {Lr_pred:.3f} D')

    for dist, xi, c in zip(PROBE_DISTANCES_D, probe_x_idx,
                            ['navy', 'purple', 'teal']):
        ax1.axvline(x_phys[xi] / D, color=c, lw=1.0, ls=':', alpha=0.7)

    error = u_mean_pred - u_mean_true
    ax2 = ax1.twinx()
    ax2.plot(x_from_cyl_D, error, color='crimson', lw=1.5, ls='--', label='Error')
    ax2.set_ylabel("Error", fontsize=11, color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')

    ax1.set_xlim((-1, 10))
    ax1.set_xlabel("x / D (downstream distance)", fontsize=11)
    ax1.set_ylabel("Time-averaged normalised u-velocity", fontsize=11)
    ax1.set_title("True over U-Net Predicted Centreline Profile", fontsize=11)
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')

    fig.suptitle(f"Wake / Recirculation Length Analysis\nTarget Lr = {LR_TARGET:.2f} D",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _savefig("Recirculation_True_over_Predicted.png", dpi=FIGURE_DPI_HIGH)


# ── 7. Three-probe temporal evolution with recirculation threshold ────────────

def plot_temporal_probes(y_test: np.ndarray, y_pred: np.ndarray,
                          wake: dict, flow_min: float, flow_max: float):
    """
    3-panel time series at probe locations with u_phys=0 threshold shading.

    Parameters
    ----------
    wake      : dict  from evaluate.compute_wake_geometry()
    flow_min / flow_max : scalars  from preprocessing (used to compute threshold)
    """
    probe_x_idx = wake["probe_x_idx"]
    x_phys      = wake["x_phys"]
    y_lower     = wake["y_lower"]
    y_upper     = wake["y_upper"]

    # Normalised value corresponding to physical u = 0
    u_zero_norm = (0.0 - flow_min) / (flow_max - flow_min)

    true_u_probes = [
        0.5 * (y_test[:, xi, y_lower, 0] + y_test[:, xi, y_upper, 0])
        for xi in probe_x_idx
    ]
    pred_u_probes = [
        0.5 * (y_pred[:, xi, y_lower, 0] + y_pred[:, xi, y_upper, 0])
        for xi in probe_x_idx
    ]
    time_steps = np.arange(y_test.shape[0])

    probe_titles = [
        f"P1 — {PROBE_DISTANCES_D[0]:.1f}D downstream  "
        f"(x-idx={probe_x_idx[0]})  |  INSIDE recirculation bubble",
        f"P2 — {PROBE_DISTANCES_D[1]:.1f}D downstream  "
        f"(x-idx={probe_x_idx[1]})  |  Near reattachment (Lr ≈ {LR_TARGET} D)",
        f"P3 — {PROBE_DISTANCES_D[2]:.1f}D downstream  "
        f"(x-idx={probe_x_idx[2]})  |  Established far wake",
    ]
    colours = ['#d62728', '#9467bd', '#17becf']

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    for ax, true_u, pred_u, title, col in zip(
            axes, true_u_probes, pred_u_probes, probe_titles, colours):

        ax.plot(time_steps, true_u, color=col, lw=1.8, label='True u')
        ax.plot(time_steps, pred_u, color=col, lw=1.2, ls='--',
                alpha=0.85, label='Predicted u')

        ax.axhline(u_zero_norm, color='black', lw=1.2, ls=':',
                   label=f'u_phys=0 (norm={u_zero_norm:.3f})')
        ax.fill_between(time_steps,
                        np.minimum(true_u, u_zero_norm), u_zero_norm,
                        where=(true_u < u_zero_norm),
                        alpha=0.15, color='red', label='Reversed flow')

        ax.set_ylabel("Normalised u", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        ax.margins(x=0)

    axes[-1].set_xlabel("Time Step (test-set frame index)", fontsize=11)
    fig.suptitle(
        "Temporal u-Velocity at 3 Wake Probes with Recirculation Threshold\n"
        f"{PROBE_DISTANCES_D[0]}D | {PROBE_DISTANCES_D[1]}D | "
        f"{PROBE_DISTANCES_D[2]}D downstream — Lr target = {LR_TARGET} D",
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    _savefig("Temporal_Probes_Recirculation_Threshold.png", dpi=FIGURE_DPI_LOW)


# ── CLI entry point ───────────────────────────────────────────────────────────

def run_all_plots():
    """
    Regenerate all figures from pre-saved .npy arrays in OUTPUT_DIR.
    Run this after evaluate.py has been executed.
    """
    print("[visualize] Loading saved arrays …")
    y_test = np.load(os.path.join(OUTPUT_DIR, "y_test.npy"))
    y_pred = np.load(os.path.join(OUTPUT_DIR, "y_pred.npy"))
    stats  = np.load(os.path.join(OUTPUT_DIR, "flow_stats.npy"))
    flow_min, flow_max = float(stats[0]), float(stats[1])

    from evaluate import compute_vorticity_metrics, compute_wake_geometry

    vort_metrics = compute_vorticity_metrics(y_test, y_pred)
    wake         = compute_wake_geometry(y_test, y_pred)

    plot_velocity_comparison(y_test, y_pred)
    plot_absolute_error(y_test, y_pred)
    plot_temporal_evolution(y_test, y_pred)
    plot_vorticity(vort_metrics)
    plot_recirculation(wake)
    plot_temporal_probes(y_test, y_pred, wake, flow_min, flow_max)
    print("[visualize] All figures saved.")


if __name__ == "__main__":
    run_all_plots()
