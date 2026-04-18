"""
train.py
========
Train (or resume) the U-Net surrogate model for cylinder wake prediction.

Usage
-----
    python train.py                   # train from scratch
    python train.py --resume          # continue from saved checkpoint
    python train.py --epochs 40       # override epoch count
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless-safe backend
import matplotlib.pyplot as plt

from data_loader import load_and_prepare
from model import build_unet
from config import (
    MODEL_PATH, OUTPUT_DIR,
    EPOCHS, BATCH_SIZE, VAL_SPLIT,
    FIGURE_DPI_LOW,
)


def train(epochs: int = EPOCHS, resume: bool = False):
    """
    Load data → build / load model → fit → save weights + loss curve.

    Parameters
    ----------
    epochs : int   number of training epochs
    resume : bool  if True, load existing weights before fitting
    """
    # ── Data ──────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, flow_min, flow_max = load_and_prepare()
    np.save(os.path.join(OUTPUT_DIR, "flow_stats.npy"),
            np.array([flow_min, flow_max]))
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_unet(X_train.shape[1:])
    model.summary()

    if resume and os.path.exists(MODEL_PATH):
        model.load_weights(MODEL_PATH)
        print(f"[train] Resumed weights from {MODEL_PATH}")

    # ── Training ──────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_split=VAL_SPLIT,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # ── Save model ────────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"[train] Model saved → {MODEL_PATH}")

    # ── Loss curve ────────────────────────────────────────────────────────────
    _plot_loss(history)

    # Final summary
    print(f"\n[train] Final Train Loss      : {history.history['loss'][-1]:.6f}")
    print(f"[train] Final Validation Loss : {history.history['val_loss'][-1]:.6f}")
    print(f"[train] Loss gap              : "
          f"{history.history['val_loss'][-1] - history.history['loss'][-1]:.6f}")

    return model, history


def _plot_loss(history):
    """Save a log-scale loss-vs-epoch plot."""
    plt.figure(figsize=(10, 7))
    plt.plot(history.history['loss'],     marker='o', label='Train Loss')
    plt.plot(history.history['val_loss'], marker='s', label='Validation Loss')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (log scale)')
    plt.title('U-Net Training Loss vs Epoch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "Loss_vs_Epoch_LogScale.png")
    plt.savefig(out, dpi=FIGURE_DPI_LOW)
    plt.close()
    print(f"[train] Loss curve saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the U-Net cylinder wake model")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from an existing saved model")
    args = parser.parse_args()

    train(epochs=args.epochs, resume=args.resume)
