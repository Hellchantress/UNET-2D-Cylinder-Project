"""
main.py
=======
End-to-end pipeline: train → evaluate → visualise.

Usage
-----
    python main.py                        # full run (train + evaluate + plot)
    python main.py --skip-train           # evaluate + plot only (use saved model)
    python main.py --epochs 40            # override epoch count
    python main.py --skip-train --plot-only  # regenerate plots from saved .npy files
"""

import argparse
import os
import numpy as np

from config import OUTPUT_DIR, MODEL_PATH


def main():
    parser = argparse.ArgumentParser(
        description="U-Net surrogate model for 2-D cylinder wake prediction"
    )
    parser.add_argument("--skip-train",  action="store_true",
                        help="Skip training; load weights from MODEL_PATH")
    parser.add_argument("--plot-only",   action="store_true",
                        help="Re-generate plots from saved .npy arrays only")
    parser.add_argument("--epochs",      type=int, default=None,
                        help="Override EPOCHS in config.py")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume training from an existing checkpoint")
    args = parser.parse_args()

    # ── Plot-only shortcut ────────────────────────────────────────────────────
    if args.plot_only:
        from visualize import run_all_plots
        run_all_plots()
        return

    # ── Training ──────────────────────────────────────────────────────────────
    if not args.skip_train:
        from train import train
        from config import EPOCHS
        epochs = args.epochs if args.epochs else EPOCHS
        train(epochs=epochs, resume=args.resume)
    else:
        print(f"[main] Skipping training — using saved model at {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No saved model found at '{MODEL_PATH}'. "
                "Run without --skip-train first."
            )

    # ── Evaluation ────────────────────────────────────────────────────────────
    from evaluate import evaluate
    y_test, y_pred, vel_metrics, vort_metrics, wake = evaluate()

    # Load normalisation stats for threshold calculation
    stats = np.load(os.path.join(OUTPUT_DIR, "flow_stats.npy"))
    flow_min, flow_max = float(stats[0]), float(stats[1])

    # ── Visualisation ─────────────────────────────────────────────────────────
    from visualize import (
        plot_velocity_comparison,
        plot_absolute_error,
        plot_temporal_evolution,
        plot_vorticity,
        plot_recirculation,
        plot_temporal_probes,
    )

    plot_velocity_comparison(y_test, y_pred)
    plot_absolute_error(y_test, y_pred)
    plot_temporal_evolution(y_test, y_pred)
    plot_vorticity(vort_metrics)
    plot_recirculation(wake)
    plot_temporal_probes(y_test, y_pred, wake, flow_min, flow_max)

    print(f"\n[main] ✓ Pipeline complete.  All outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
