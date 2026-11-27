from __future__ import annotations

import numpy as np
import pytest

from pta.weights import drive_voltages, quantize, spectral_norm, target_matrices, tile_weights


def test_target_matrices(cfg):
    matrices = target_matrices(cfg)
    assert matrices.shape == (cfg.K, cfg.N, cfg.N)
    for k in range(cfg.K):
        assert spectral_norm(matrices[k]) == pytest.approx(1.0)


def test_quantize_round_trip():
    matrix = np.array([[0.5, -0.25], [0.125, -1.0]])
    quantized, peak = quantize(matrix, bits=8)
    assert peak == pytest.approx(1.0)
    np.testing.assert_allclose(quantized * peak, matrix, atol=1.0 / 127.0)


def test_tile_weights(cfg):
    weights = np.random.default_rng(0).uniform(-1, 1, (cfg.K * cfg.N, cfg.N))
    tiles, scales = tile_weights(weights, cfg, bits=8)
    assert tiles.shape == (cfg.K, cfg.N, cfg.N)
    assert scales.shape == (cfg.K,)
    assert np.all(scales > 0)
    for k in range(cfg.K):
        assert spectral_norm(tiles[k]) == pytest.approx(1.0, rel=1e-9)


def test_tile_weights_rejects_small_layer(cfg):
    with pytest.raises(ValueError):
        tile_weights(np.ones((cfg.N, cfg.N)), cfg, bits=8)


def test_drive_voltage_encodes_signed_input(cfg):
    inputs = np.linspace(-1.0, 1.0, 9).reshape(3, 3)
    voltages = drive_voltages(inputs, cfg)
    recovered = np.sin(np.pi * voltages / (2.0 * cfg.v_pi))
    np.testing.assert_allclose(recovered, inputs, atol=1e-12)
