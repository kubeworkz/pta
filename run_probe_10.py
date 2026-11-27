import sys
sys.path.insert(0, "src")
import tidy3d as td
import tidy3d.web as web
from pta.config import AcceleratorConfig
import numpy as np

cfg = AcceleratorConfig()
y0 = cfg.port_y(0)
freqs = cfg.freqs_hz
distance_um = cfg.design_lx + 2.0
x_source = -distance_um/2
x_monitor = distance_um/2

source = td.ModeSource(center=(x_source,y0,0.0), size=(0.0,2.5,cfg.t_sin+1.5),
    source_time=td.GaussianPulse(freq0=float(np.mean(freqs)), fwidth=0.15*np.mean(freqs)),
    direction="+", mode_spec=td.ModeSpec(num_modes=1), name="input")
out = td.ModeMonitor(center=(x_monitor,y0,0.0), size=(0.0,2.0,cfg.t_sin+1.5),
    freqs=freqs, mode_spec=td.ModeSpec(num_modes=1), name="out_0")
refl = td.ModeMonitor(center=(x_source-0.4,y0,0.0), size=(0.0,2.0,cfg.t_sin+1.5),
    freqs=freqs, mode_spec=td.ModeSpec(num_modes=1), name="refl_0")
sin = td.Structure(geometry=td.Box(center=(0.0,y0,0.0), size=(distance_um+4.0,cfg.wg_width,cfg.t_sin)),
    medium=td.Medium(permittivity=cfg.n_sin**2), name="SiN")
sim = td.Simulation(center=(0.0,y0,0.0), size=(24.0,8.0,2.8),
    run_time=500/np.mean(freqs), grid_spec=td.GridSpec.auto(min_steps_per_wvl=25, wavelength=1.55),
    structures=[sin], sources=[source], monitors=[out,refl],
    medium=td.Medium(permittivity=cfg.n_sio2**2), boundary_spec=td.BoundarySpec.all_sides(td.PML()),
    plot_length_units="um", shutoff=0.00001, subpixel=td.SubpixelSpec(dielectric=td.PolarizedAveraging()))
job = web.Job(simulation=sim, task_name="loss_probe_probe_10_SiN_only")
print("estimate:", job.estimate_cost())
job.upload()
job.start()
data = job.load(path="artifacts/tidy3d/ta_rm_test.hdf5")
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
print("DONE")
