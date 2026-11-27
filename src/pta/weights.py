"""Signed weight matrices for the tensor core.

Two sources of targets are supported:

* :func:`target_matrices` - deterministic synthetic matrices used for the golden build and
  for regression tests.
* :func:`tile_weights` - a real trained layer, quantized and split into ``K`` ``N x N``
  hardware tiles, each spectrally normalized so the passive core can realize it
  (``||W_tile||_2 <= 1``). The per-tile scale is applied digitally after the ADC.
"""

from __future__ import annotations

import numpy as np

from .config import AcceleratorConfig


def spectral_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(matrix, ord=2))


def target_matrices(cfg: AcceleratorConfig) -> np.ndarray:
    """Return ``K`` signed target matrices of shape ``(K, N, N)`` with unit spectral norm."""
    rng = np.random.default_rng(cfg.seed)
    matrices = rng.uniform(-1.0, 1.0, size=(cfg.K, cfg.N, cfg.N))
    norms = np.linalg.norm(matrices, ord=2, axis=(1, 2), keepdims=True)
    return matrices / norms


def quantize(matrix: np.ndarray, bits: int = 8) -> tuple[np.ndarray, float]:
    """Symmetric uniform quantization; returns the quantized matrix and its de-scaling factor."""
    if bits < 2:
        raise ValueError("bits must be >= 2")
    q_max = float(2 ** (bits - 1) - 1)
    peak = float(np.max(np.abs(matrix)))
    if peak <= 0.0:
        return np.zeros_like(matrix), 1.0
    return np.round(matrix / peak * q_max) / q_max, peak


def tile_weights(
    weights: np.ndarray, cfg: AcceleratorConfig, bits: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Split a trained layer into ``K`` passive ``N x N`` tiles.

    Args:
        weights: real matrix of shape ``(out_dim, in_dim)``.
        cfg: hardware configuration supplying ``N`` and ``K``.
        bits: weight quantization width.

    Returns:
        ``(tiles, scales)`` with ``tiles`` of shape ``(K, N, N)`` and ``scales`` of shape
        ``(K,)``. Multiply the measured photocurrents by ``scales`` after the ADC to
        recover the unnormalized layer output.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 2:
        raise ValueError("weights must be a 2D matrix")
    n, k = cfg.N, cfg.K

    quantized, peak = quantize(weights, bits)
    pad_rows = (-quantized.shape[0]) % n
    pad_cols = (-quantized.shape[1]) % n
    padded = np.pad(quantized, ((0, pad_rows), (0, pad_cols)), mode="constant")
    n_tiles = (padded.shape[0] // n) * (padded.shape[1] // n)
    if n_tiles < k:
        raise ValueError(
            f"layer of shape {weights.shape} yields {n_tiles} tiles of {n}x{n}, "
            f"fewer than K={k} comb lines"
        )

    tiles = np.zeros((k, n, n), dtype=float)
    scales = np.zeros(k, dtype=float)
    index = 0
    for row in range(padded.shape[0] // n):
        for col in range(padded.shape[1] // n):
            if index >= k:
                break
            block = padded[row * n : (row + 1) * n, col * n : (col + 1) * n]
            sigma = max(spectral_norm(block), 1e-12)
            tiles[index] = block / sigma
            scales[index] = sigma * peak
            index += 1

    return tiles, scales


def scale_inputs(inputs: np.ndarray, peak: float | None = None) -> tuple[np.ndarray, float]:
    """Normalize a batch of signed input vectors into ``[-1, 1]`` for the MZM drive."""
    inputs = np.asarray(inputs, dtype=float)
    peak = float(np.max(np.abs(inputs))) if peak is None else float(peak)
    if peak <= 0.0:
        return inputs.copy(), 1.0
    return inputs / peak, peak


def drive_voltages(inputs: np.ndarray, cfg: AcceleratorConfig) -> np.ndarray:
    """Map signed inputs in ``[-1, 1]`` to null-biased push-pull drive voltages.

    A null-biased push-pull MZM imparts ``E_out = sin(pi V / (2 V_pi))``, so inverting that
    relation yields the bipolar drive that encodes the sign in the optical phase.
    """
    normalized = np.clip(np.asarray(inputs, dtype=float), -1.0, 1.0)
    return (2.0 * cfg.v_pi / np.pi) * np.arcsin(normalized)
