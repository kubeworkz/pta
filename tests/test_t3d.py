from __future__ import annotations

import numpy as np
import pytest
import tidy3d as td

from pta.core import build_core
from pta.t3d import (
    build_inverse_design,
    build_straight_validation_sim,
    build_verification_sim,
    design_beta,
    design_initial_density,
    design_spec,
    design_tasks,
    validate_design,
    write_straight_validation_spec,
)


def _box(sim, name):
    structure = next(structure for structure in sim.structures if structure.name == name)
    assert isinstance(structure.geometry, td.Box)
    return structure.geometry


def _x_bounds(box):
    (xmin, _, _), (xmax, _, _) = box.bounds
    return xmin, xmax


def _y_bounds(box):
    (_, ymin, _), (_, ymax, _) = box.bounds
    return ymin, ymax


def test_straight_guide_covers_the_source_and_monitor(cfg):
    sim = build_straight_validation_sim(cfg)
    source = sim.sources[0]
    out = next(monitor for monitor in sim.monitors if monitor.name == "out_0")

    guide_xmin, guide_xmax = _x_bounds(_box(sim, "SiN"))
    guide_half = cfg.design_lx / 2 + max(cfg.stub_len, 1.5)
    assert guide_xmin == pytest.approx(-guide_half)
    assert guide_xmax == pytest.approx(guide_half)
    assert guide_xmin < source.center[0] < guide_xmax
    assert guide_xmin < out.center[0] < guide_xmax
    lead = guide_half - cfg.design_lx / 2 - 1.0
    assert source.center[0] - guide_xmin == pytest.approx(lead)
    assert guide_xmax - out.center[0] == pytest.approx(lead)

    slab_xmin, slab_xmax = _x_bounds(_box(sim, "TFLN"))
    assert slab_xmin <= guide_xmin
    assert slab_xmax >= guide_xmax

    # Lane 0 is at y = -6 um; the slab must follow it, not stay at the chip centre.
    slab_ymin, slab_ymax = _y_bounds(_box(sim, "TFLN"))
    assert slab_ymin < source.center[1] < slab_ymax
    assert slab_ymin < out.center[1] < slab_ymax


def test_straight_validation_uses_core_source_and_monitor_settings(pipeline_result):
    cfg = pipeline_result.cfg
    straight = build_straight_validation_sim(cfg)
    core = build_verification_sim(cfg, pipeline_result.artifacts["core_gds"], 0)

    straight_source = straight.sources[0]
    core_source = core.sources[0]
    assert straight_source.center == core_source.center
    assert straight_source.size == core_source.size
    assert straight_source.direction == core_source.direction
    assert straight_source.name == core_source.name
    assert straight_source.source_time.model_dump() == core_source.source_time.model_dump()
    assert straight_source.mode_spec.model_dump() == core_source.mode_spec.model_dump()

    for name in ("out_0", "refl_0"):
        straight_monitor = next(
            monitor for monitor in straight.monitors if monitor.name == name
        )
        core_monitor = next(monitor for monitor in core.monitors if monitor.name == name)
        assert straight_monitor.center == core_monitor.center
        # The straight validator intentionally widens the output/reflection monitor
        # to 8.0 um in y so the TFLN-guided mode is fully captured (the core sim
        # keeps 2.0 um per-lane monitors). Direction, mode_spec, freqs and names
        # are still shared; only the monitor y-size differs between the two sims.
        assert straight_monitor.size[0] == core_monitor.size[0]
        assert straight_monitor.size[2] == core_monitor.size[2]
        assert straight_monitor.size[1] == 8.0
        assert core_monitor.size[1] == 2.0
        assert straight_monitor.name == core_monitor.name
        assert straight_monitor.mode_spec.model_dump() == core_monitor.mode_spec.model_dump()
        np.testing.assert_allclose(straight_monitor.freqs, core_monitor.freqs)


def test_straight_validation_spec_round_trips(tmp_path, cfg):
    path = write_straight_validation_spec(cfg, tmp_path / "tidy3d_validate_straight.json")
    text = path.read_text(encoding="utf-8")

    assert text.isascii()
    sim = td.Simulation.model_validate_json(text)
    assert isinstance(sim, td.Simulation)
    assert {monitor.name for monitor in sim.monitors} == {"out_0", "refl_0"}


def test_inverse_design_is_one_sourced_simulation_per_line_and_input(cfg):
    design, objective = build_inverse_design(cfg, build_core(cfg).weights)
    tasks = design_tasks(cfg)

    assert callable(objective)
    assert len(design.simulations) == len(tasks) == cfg.K * cfg.N
    for (line, index), simulation in zip(tasks, design.simulations):
        # A source-free design simulation is rejected at upload, so every task carries the
        # mode source for its own input port at its own comb line.
        assert len(simulation.sources) == 1
        source = simulation.sources[0]
        assert source.center[1] == pytest.approx(cfg.port_y(index))
        assert source.source_time.freq0 == pytest.approx(cfg.freqs_hz[line])

        # The metric validator accepts one frequency per job, so every monitor is sampled
        # at the task's line only.
        assert len(simulation.monitors) == cfg.N
        for monitor in simulation.monitors:
            np.testing.assert_allclose(monitor.freqs, [cfg.freqs_hz[line]])

    # The access stubs must reach the design-region edge, or the design is not connected.
    simulation = design.simulations[0]
    stub = next(s for s in simulation.structures if s.name == "SiN_stub_0_in")
    (xmin, _, _), (xmax, _, _) = stub.geometry.bounds
    assert xmax == pytest.approx(-cfg.design_lx / 2)
    assert xmin == pytest.approx(-cfg.design_lx / 2 - cfg.stub_len)
    assert xmin < simulation.sources[0].center[0] < xmax


def test_design_initial_density_keeps_every_port_connected(cfg):
    from pta import blocks, polygons

    density = design_initial_density(cfg)
    assert density.shape == (cfg.nx, cfg.ny)
    assert 0.0 <= density.min() <= density.max() <= 1.0

    # The strict ingested-core limit: an EM core must present material at every port
    # within 0.6 um, so the seed has to survive the polygon build without a rescue stub.
    core = build_core(cfg, density=density, backend="tidy3d")
    polygons_um = polygons.density_to_polygons(density, cfg)
    _, landings = blocks.freeform_core(cfg, core, polygons_um, cfg.freqs_hz)

    assert len(landings) == 2 * cfg.N
    assert max(landings.values()) <= cfg.stub_landing_em_max_um


def test_design_region_matches_the_spec_filter_and_projection(cfg):
    import tidy3d.plugins.invdes as tdi

    design, _ = build_inverse_design(cfg, build_core(cfg).weights, beta=design_beta(3))
    (transformation,) = design.design_region.transformations
    (penalty,) = design.design_region.penalties

    assert isinstance(transformation, tdi.FilterProject)
    assert transformation.radius == pytest.approx(cfg.drc_min_feature)
    assert transformation.beta == pytest.approx(design_beta(3))
    assert isinstance(penalty, tdi.ErosionDilationPenalty)
    assert penalty.length_scale == pytest.approx(cfg.drc_freeform_um)

    spec = design_spec(cfg, build_core(cfg).weights)
    assert spec["optimizer"]["filter"] == "conic"
    assert spec["optimizer"]["filter_radius_um"] == pytest.approx(transformation.radius)
    assert spec["optimizer"]["projection"] == "ramp_tanh"
    assert spec["optimizer"]["beta_schedule"][2] == pytest.approx(design_beta(3))
    assert spec["optimizer"]["penalty"]["length_scale_um"] == pytest.approx(penalty.length_scale)
    assert spec["objective"]["type"] == "phase_invariant_overlap"


def _objective_batch(cfg, weights, tasks, names, scale):
    """Synthetic per-task mode amplitudes: ``measured = scale * W[k][:, i] e^{i phi}``."""
    import xarray as xr

    batch = {}
    for (line, index), name in zip(tasks, names):
        target = weights[line][:, index] * np.exp(1j * cfg.phi_ref[line])
        datasets = {}
        for out in range(cfg.N):
            amps = xr.DataArray(
                (scale * target[out]).reshape(1, 1, -1),
                coords={
                    "direction": ["+"],
                    "mode_index": [0],
                    "f": [cfg.freqs_hz[line]],
                },
            )
            datasets[f"out_{out}"] = xr.Dataset({"amps": amps})
        batch[name] = datasets
    return batch


def test_design_objective_is_phase_and_gain_invariant(cfg):
    from pta.t3d import design_objective

    weights = build_core(cfg).weights
    tasks = design_tasks(cfg)
    names = [f"synthetic_{index}" for index in range(len(tasks))]
    objective = design_objective(cfg, weights, tasks=tasks, task_names=names)

    # A perfect kernel shape at an arbitrary complex gain must score full overlap; the
    # receiver calibrates that gain and phase away, so the objective must not punish it.
    batch = _objective_batch(cfg, weights, tasks, names, 0.3 * np.exp(1.2j))

    assert float(objective(batch)) == pytest.approx(len(tasks), abs=1e-6)


def test_design_objective_rejects_a_dark_mask(cfg):
    from pta.t3d import design_objective

    weights = build_core(cfg).weights
    tasks = design_tasks(cfg)
    names = [f"synthetic_{index}" for index in range(len(tasks))]
    objective = design_objective(cfg, weights, tasks=tasks, task_names=names)

    # Zero transmission is perfectly "consistent" and must not be a free optimum.
    batch = _objective_batch(cfg, weights, tasks, names, 0.0)

    assert float(objective(batch)) < 0.0


def test_projected_initial_density_keeps_every_port_connected(cfg):
    from pta import blocks, polygons

    design, _ = build_inverse_design(cfg, build_core(cfg).weights)
    params = design_initial_density(cfg)[:, :, None]
    density = np.asarray(design.design_region.material_density(params))[..., 0]

    # The filter and projection are what the solver actually sees, so the connectivity
    # check has to hold for the projected map, not the raw parameters.
    core = build_core(cfg, density=density, backend="tidy3d")
    polygons_um = polygons.density_to_polygons(density, cfg)
    _, landings = blocks.freeform_core(cfg, core, polygons_um, cfg.freqs_hz)

    assert len(landings) == 2 * cfg.N
    assert max(landings.values()) <= cfg.stub_landing_em_max_um


def test_validate_design_names_the_source_free_task():
    import tidy3d.plugins.invdes as tdi

    region = tdi.TopologyDesignRegion(
        size=(2.0, 2.0, 0.2), center=(0.0, 0.0, 0.0), eps_bounds=(1.0, 4.0), pixel_size=0.2
    )
    simulation = td.Simulation(
        size=(4.0, 4.0, 1.0),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.2),
        boundary_spec=td.BoundarySpec.all_sides(td.PML()),
        sources=[],
    )
    design = tdi.InverseDesign(
        design_region=region, simulation=simulation, task_name="unit_source_free"
    )

    with pytest.raises(ValueError, match="unit_source_free"):
        validate_design(design)


def test_pipeline_writes_straight_validation_spec(pipeline_result):
    path = pipeline_result.artifacts["t3d_validate"]
    assert path.exists() and path.stat().st_size > 0
    assert path.name == "tidy3d_validate_straight.json"
