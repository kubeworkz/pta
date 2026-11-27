from __future__ import annotations

import gdstk
import numpy as np
import pytest

from pta import blocks
from pta.core import build_core, fidelity, physics_report, surrogate_density


def test_core_shape(cfg):
    core = build_core(cfg)
    assert core.s_matrix.shape == (cfg.K, 2 * cfg.N, 2 * cfg.N)
    assert core.weights.shape == (cfg.K, cfg.N, cfg.N)
    assert core.transmission().shape == (cfg.K, cfg.N, cfg.N)


def test_reciprocity_and_passivity(cfg):
    report = physics_report(build_core(cfg))
    assert report["reciprocal"] is True
    assert report["passive"] is True
    for line in report["lines"]:
        assert line["reciprocity_error"] < 1e-9
        assert line["max_singular_value"] <= 1.0 + 1e-9


def test_target_matrix_spectral_norm(cfg):
    core = build_core(cfg)
    for k in range(cfg.K):
        assert np.linalg.norm(core.weights[k], ord=2) == pytest.approx(1.0, rel=1e-9)


def test_fidelity_tracks_phase_error(cfg):
    perfect = fidelity(build_core(cfg, phase_error_rad=0.0))
    noisy = fidelity(build_core(cfg, phase_error_rad=0.25))
    assert perfect["worst_relative_error"] < 1e-9
    assert noisy["worst_relative_error"] > 0.05
    for line in perfect["per_line"]:
        assert line["amplitude_scale"] > 0.5


def test_core_stub_lands_on_an_island(cfg):
    """The stub is extended until it shares real area with the pattern."""
    y = cfg.port_y(2)
    island = gdstk.rectangle((-8.5, y - 0.5), (-6.0, y + 0.5))
    stub, depth = blocks._stub_landing(
        cfg, [island], y, -1.0, "IN_2", cfg.stub_landing_max_um
    )
    (_, y0), (x1, y1) = stub.bounding_box()
    assert x1 > -8.5, "the stub reaches into the island"
    assert y1 - y0 == pytest.approx(cfg.wg_width, abs=1e-9)
    assert depth == pytest.approx(0.7, abs=0.11)


def test_core_stub_must_find_an_island(cfg):
    """A port with no material in reach fails the build instead of floating."""
    y = cfg.port_y(2)
    far = gdstk.rectangle((0.0, y - 0.5), (1.0, y + 0.5))
    with pytest.raises(ValueError, match="floating"):
        blocks._stub_landing(cfg, [far], y, -1.0, "IN_2", cfg.stub_landing_max_um)


def test_ingested_core_uses_a_tight_stub_limit(cfg):
    """A real core cannot hide a multi-micron lead behind the surrogate allowance."""
    assert blocks.stub_landing_limit(cfg, build_core(cfg)) == cfg.stub_landing_max_um
    assert (
        blocks.stub_landing_limit(cfg, build_core(cfg, backend="fdtd"))
        == cfg.stub_landing_em_max_um
    )
    assert cfg.stub_landing_em_max_um < 0.5 * cfg.stub_landing_max_um
    y = cfg.port_y(2)
    island = gdstk.rectangle((-8.0, y - 0.5), (-6.0, y + 0.5))  # 1 um inside the boundary
    with pytest.raises(ValueError, match="0.6 um"):
        blocks._stub_landing(cfg, [island], y, -1.0, "IN_2", cfg.stub_landing_em_max_um)


def test_surrogate_density(cfg):
    density = surrogate_density(cfg)
    assert density.shape == (cfg.nx, cfg.ny)
    assert 0.0 <= density.min() < 0.2
    assert 0.8 < density.max() <= 1.0
    # about half the area is material, which keeps the mask balanced
    assert 0.3 < float((density >= 0.5).mean()) < 0.7


def test_npz_round_trip(cfg, tmp_path):
    core = build_core(cfg)
    path = core.save(tmp_path / "core.npz")
    loaded = type(core).from_npz(path)
    np.testing.assert_allclose(loaded.s_matrix, core.s_matrix)
    np.testing.assert_allclose(loaded.weights, core.weights)
    assert loaded.backend == core.backend
    assert loaded.metadata()["N"] == cfg.N
