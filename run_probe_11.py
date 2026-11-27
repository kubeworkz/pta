import sys
sys.path.insert(0, "src")
import tidy3d as td
import tidy3d.web as web
from pta.config import AcceleratorConfig
import numpy as np

cfg = AcceleratorConfig()
y0 = cfg.port_y(0)            # -6.0 um, lane 0
freqs = cfg.freqs_hz
distance_um = cfg.design_lx + 2.0   # 20.0 um
x_source = -distance_um/2
x_monitor = distance_um/2

# Source kept as core spec: y-size 2.5, z-size t_sin+1.5, centred on lane 0.
source = td.ModeSource(center=(x_source,y0,0.0), size=(0.0,2.5,cfg.t_sin+1.5),
    source_time=td.GaussianPulse(freq0=float(np.mean(freqs)), fwidth=0.15*np.mean(freqs)),
    direction="+", mode_spec=td.ModeSpec(num_modes=1), name="input")

# Output monitor widened in y to 8.0 um to capture the full TFLN-guided mode
# (the original 2.0 um monitor sits inside the mode's evanescent tails and under-
# reports the guided power by ~17%, which is the 0.826 vs ~1.0 discrepancy we
# are isolating). TFLN slab widened to match so the slab is not extruded through
# the monitor plane.
out = td.ModeMonitor(center=(x_monitor,y0,0.0), size=(0.0,8.0,cfg.t_sin+1.5),
    freqs=freqs, mode_spec=td.ModeSpec(num_modes=1), name="out_0")
refl = td.ModeMonitor(center=(x_source-0.4,y0,0.0), size=(0.0,8.0,cfg.t_sin+1.5),
    freqs=freqs, mode_spec=td.ModeSpec(num_modes=1), name="refl_0")

sin = td.Structure(geometry=td.Box(center=(0.0,y0,0.0), size=(distance_um+4.0,cfg.wg_width,cfg.t_sin)),
    medium=td.Medium(permittivity=cfg.n_sin**2), name="SiN")
tfln = td.Structure(geometry=td.Box(center=(0.0,y0,cfg.t_sin/2+cfg.t_gap+cfg.t_tfln/2),
    size=(distance_um+6.0,8.0,cfg.t_tfln)),
    medium=td.AnisotropicMedium(xx=td.Medium(permittivity=cfg.n_tfln_o**2),
        yy=td.Medium(permittivity=cfg.n_tfln_o**2),
        zz=td.Medium(permittivity=cfg.n_tfln_e**2)), name="TFLN")

sim = td.Simulation(center=(0.0,y0,0.0), size=(24.0,12.0,2.8),
    run_time=500/np.mean(freqs), grid_spec=td.GridSpec.auto(min_steps_per_wvl=25, wavelength=1.55),
    structures=[sin,tfln], sources=[source], monitors=[out,refl],
    medium=td.Medium(permittivity=cfg.n_sio2**2), boundary_spec=td.BoundarySpec.all_sides(td.PML()),
    plot_length_units="um", shutoff=0.00001, subpixel=td.SubpixelSpec(dielectric=td.PolarizedAveraging()))

job = web.Job(simulation=sim, task_name="loss_probe_probe_11_wide_monitor")
print("estimate:", job.estimate_cost())
job.upload()
job.start()
data = job.load(path="artifacts/tidy3d/loss_probe_probe_11.hdf5")
for mname in ["out_0","refl_0"]:
    mon = data[mname]
    fwd = mon.amps.sel(direction="+", mode_index=0)
    bwd = mon.amps.sel(direction="-", mode_index=0)
    fwd_pow = np.abs(fwd.values)**2
    bwd_pow = np.abs(bwd.values)**2
    nc = np.asarray(mon.n_complex.values)
    print(f"[{mname}] f(THz)={[float(fr/1e12) for fr in mon.amps.coords['f'].values]}")
    print(f"   |a_fwd|^2 = {[float(v) for v in fwd_pow]} mean={float(np.mean(fwd_pow)):.4f}")
    print(f"   |a_bwd|^2 = {[float(v) for v in bwd_pow]}")
    print(f"   n_complex.real = {[float(v.real) for v in nc.flatten()]} n_complex.imag = {[float(v.imag) for v in nc.flatten()]}")
mon = data["out_0"]
fwd = mon.amps.sel(direction="+", mode_index=0)
omega = 2*np.pi*np.asarray(mon.amps.coords['f'].values)
phase = np.unwrap(np.angle(fwd.values))
slope = float(np.polyfit(omega - omega.mean(), phase.flatten(), 1)[0])
ng = 299792458.0 * slope / (20.0e-6)
print("fitted n_g (20 um) =", round(ng,4))
print("DONE")
