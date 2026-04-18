"""
model.py
========
U-Net convolutional encoder-decoder for next-step flow-field prediction.

Architecture:
  Encoder  : 3 stages  (64 → 128 → 256 filters, each stage 2×Conv2D + MaxPool)
  Bottleneck: 512 filters
  Decoder  : 3 stages  (UpSampling2D + skip concatenation + Conv2D)
  Output   : 1×1 Conv2D → 2 channels (u, v)

Skip connections from each encoder stage are concatenated with the
corresponding decoder stage (standard U-Net design).
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_unet(input_shape: tuple) -> tf.keras.Model:
    """
    Construct and return the compiled U-Net model.

    Parameters
    ----------
    input_shape : tuple  (H, W, C)  — spatial dimensions and channels (2 for u, v)

    Returns
    -------
    model : tf.keras.Model  (compiled, Adam + MSE)
    """
    inputs = layers.Input(shape=input_shape)

    # ── Encoder ──────────────────────────────────────────────────────────────
    # Stage 1
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    # Stage 2
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    # Stage 3
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    # ── Bottleneck ────────────────────────────────────────────────────────────
    b = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)

    # ── Decoder ───────────────────────────────────────────────────────────────
    # Stage 3 ↑
    u1 = layers.UpSampling2D()(b)
    u1 = layers.concatenate([u1, c3])
    u1 = layers.Conv2D(256, 3, activation='relu', padding='same')(u1)

    # Stage 2 ↑
    u2 = layers.UpSampling2D()(u1)
    u2 = layers.concatenate([u2, c2])
    u2 = layers.Conv2D(128, 3, activation='relu', padding='same')(u2)

    # Stage 1 ↑
    u3 = layers.UpSampling2D()(u2)
    u3 = layers.concatenate([u3, c1])
    u3 = layers.Conv2D(64, 3, activation='relu', padding='same')(u3)

    # ── Output ────────────────────────────────────────────────────────────────
    outputs = layers.Conv2D(2, 1, activation='linear')(u3)

    model = models.Model(inputs=inputs, outputs=outputs, name="UNet_CylinderWake")
    model.compile(optimizer='adam', loss='mse')

    return model


if __name__ == "__main__":
    # Quick sanity-check: build model for a (160, 40, 2) field
    m = build_unet((160, 40, 2))
    m.summary()
