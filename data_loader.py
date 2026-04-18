"""
data_loader.py
==============
Loads the ETH Zürich 2-D unsteady cylinder flow VTI dataset,
extracts velocity fields, normalises, and builds (X, y) sequence pairs
for next-step prediction.

Dataset source:
    ETH Zürich Computer Graphics Laboratory
    https://cgl.ethz.ch/research/fluid_simulation/data.php
    File: cylinder2d.vti   (Re ≈ 160, 1501 time steps, dims 640×80×1501)
"""

import numpy as np
import pyvista as pv

from config import (
    DATA_PATH,
    DOWNSAMPLE_X, DOWNSAMPLE_Y,
    TRAIN_SPLIT,
)


def load_vti(path: str = DATA_PATH):
    """
    Read the VTI file and return reshaped u, v arrays.

    Returns
    -------
    u, v : np.ndarray  shape (T, Nx, Ny)
        Time-series of velocity components in (t, x, y) layout.
    dims  : tuple  original grid dimensions (Nx, Ny, T)
    """
    grid = pv.read(path)
    dims = grid.dimensions          # (640, 80, 1501)
    print(f"[data_loader] Raw grid dimensions: {dims}")

    # VTK uses Fortran-order (column-major) memory layout — order='F' is critical.
    u = grid['u'].reshape(dims, order='F')   # (Nx, Ny, T)
    v = grid['v'].reshape(dims, order='F')

    print(f"[data_loader] Reshaped u: {u.shape},  v: {v.shape}")

    # Convert (x, y, t) → (t, x, y) for ML conventions
    u = np.transpose(u, (2, 0, 1))  # (T, Nx, Ny)
    v = np.transpose(v, (2, 0, 1))

    return u, v, dims


def preprocess(u: np.ndarray, v: np.ndarray):
    """
    Spatially downsample, stack channels, and min–max normalise.

    Parameters
    ----------
    u, v : np.ndarray  shape (T, Nx, Ny)

    Returns
    -------
    flow      : np.ndarray  shape (T, Nx', Ny', 2)  normalised
    flow_min  : float  global minimum before normalisation
    flow_max  : float  global maximum before normalisation
    """
    # Stack → (T, Nx, Ny, 2)
    flow = np.stack([u, v], axis=-1)

    # Spatial downsampling
    flow = flow[:, ::DOWNSAMPLE_X, ::DOWNSAMPLE_Y, :]
    print(f"[data_loader] Downsampled flow shape: {flow.shape}")

    # Global min–max normalisation to [0, 1]
    flow_min = float(flow.min())
    flow_max = float(flow.max())
    flow = (flow - flow_min) / (flow_max - flow_min)

    return flow, flow_min, flow_max


def create_sequence_dataset(flow: np.ndarray):
    """
    Build (X[t], y[t+1]) pairs for one-step-ahead prediction.

    Parameters
    ----------
    flow : np.ndarray  shape (T, H, W, C)

    Returns
    -------
    X, y : np.ndarray  shape (T-1, H, W, C)
    """
    X = flow[:-1]
    y = flow[1:]
    return X, y


def train_test_split(X: np.ndarray, y: np.ndarray, split: float = TRAIN_SPLIT):
    """
    Chronological (non-shuffled) train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    n = int(split * len(X))
    return X[:n], X[n:], y[:n], y[n:]


def load_and_prepare(path: str = DATA_PATH):
    """
    Convenience wrapper: load → preprocess → create dataset → split.

    Returns
    -------
    X_train, X_test, y_train, y_test, flow_min, flow_max
    """
    u, v, _ = load_vti(path)
    flow, flow_min, flow_max = preprocess(u, v)
    X, y = create_sequence_dataset(flow)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    print(f"[data_loader] Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, flow_min, flow_max


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, fmin, fmax = load_and_prepare()
    print(f"flow_min = {fmin:.4f},  flow_max = {fmax:.4f}")
