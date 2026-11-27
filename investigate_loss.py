import numpy as np, sys, tidy3d as td
sys.path.insert(0, "src")
from pta.config import AcceleratorConfig, C_0
from pta.validate_straight import evaluate, _mode_amplitudes, _propagation_distance

cfg = AcceleratorConfig()
y0 = cfg.port_y(0)
freqs = cfg.freqs_hz
omega = 2*np.pi*freqs
distance_um = cfg.design_lx + 2.0  # 20 um, matches loaded result
x_source = -distance_um/2
x_monitor = distance_um/2

# We look only at what the validator *sees*: the monitor object handed to the
# loader, the forward amplitude stored in it, and the source center derived
# from the embedded simulation. We keep the forward amplitude fixed and vary
# only the monitor's y-size (the candidate normalization mismatch) so any
# change in the reported forward power is a parser/geometry effect, not FDTD
# convergence. The forward answer here is the known-good value for a lossless
# straight waveguide; Tidy3D's own normalization would make it ~1 in that case.

def make_data(monitor_y):
    source = td.ModeSource(
        center=(x_source, y0, 0.0),
        size=(0.0, 2.5, cfg.t_sin + 1.5),
        source_time=td.GaussianPulse(freq0=float(np.mean(freqs)), fwidth=0.15*np.mean(freqs)),
        direction="+",
        mode_spec=td.ModeSpec(num_modes=1),
        name="input",
    )
    out = td.ModeMonitor(
        center=(x_monitor, y0, 0.0),
        size=(0.0, monitor_y, cfg.t_sin + 1.5),
        freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=1),
        name="out_0",
    )
    refl = td.ModeMonitor(
        center=(x_source - 0.4, y0, 0.0),
        size=(0.0, monitor_y, cfg.t_sin + 1.5),
        freqs=freqs,
        mode_spec=td.ModeSpec(num_modes=1),
        name="refl_0",
    )
    sim = td.Simulation(
        center=(0.0, y0, 0.0),
        size=(distance_um + 4.0, 8.0, 2.8),
        run_time=500/np.mean(freqs),
        grid_spec=td.GridSpec.auto(min_steps_per_wvl=25, wavelength=1.55),
        # no structures: straight waveguide absent -> no scattering -> mode
        # monitor should report unity if the source/monitor pair is normalized
        # consistently. We only vary monitor_y here.
        sources=[source],
        monitors=[out, refl],
        medium=td.Medium(permittivity=cfg.n_sio2**2),
        boundary_spec=td.BoundarySpec.all_sides(td.PML()),
        plot_length_units="um",
        shutoff=0.00001,
    )
    # Inject the known forward answer directly into the monitor's stored data.
    # This isolates the validator from solver noise while still exercising the
    # parser, the distance derivation, and the phase fit.
    forward = np.full(freqs.size, 0.99, dtype=complex)
    backward = np.full(freqs.size, 1e-4 + 0.0j)
    reflection = np.full(freqs.size, 2e-4 + 0.0j)
    def md(monitor, fwd, bwd):
        return td.ModeData(
            monitor=monitor,
            amps=td.DataArray(
                np.stack([fwd, bwd], axis=1)[:, :, None],
                coords={"f": freqs, "direction": ["+", "-"], "mode_index": [0]},
                dims=("f", "direction", "mode_index"),
            ),
            n_complex=td.DataArray(
                np.ones((freqs.size, 1), dtype=complex),
                coords={"f": freqs, "mode_index": [0]},
                dims=("f", "mode_index"),
            ),
        )
    data = td.SimulationData(
        simulation=sim,
        data=(md(out, forward, backward), md(refl, np.zeros_like(forward), reflection)),
    )
    return data

print("monitor_y  forward  backward   reflection   ng     L(um)")
for y in (2.0, 2.5, 4.0, 6.0, 8.0):
    r = evaluate(make_data(y))
    print(f"  {y:4.1f}   {r.forward_power[0]:.4f}   {r.backward_power[0]:.2e}   {r.reflection_power[0]:.2e}   {r.group_index:.3f}   {r.distance_um:.1f}")
