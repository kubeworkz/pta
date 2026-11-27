import sys, time
sys.path.insert(0, "src")
import tidy3d as td
import tidy3d.web as web
import numpy as np
from pta.config import AcceleratorConfig
from pta.t3d import build_straight_validation_sim

cfg = AcceleratorConfig()
sim = build_straight_validation_sim(cfg)
out = next(m for m in sim.monitors if m.name == "out_0")
refl = next(m for m in sim.monitors if m.name == "refl_0")
print("new spec: out_0.size =", out.size, "refl_0.size =", refl.size)
print("sim.size =", sim.size, "sim.center =", sim.center)

task_name = "loss_probe_probe_12b_new_spec_retry"
job = web.Job(simulation=sim, task_name=task_name)
print("estimate:", job.estimate_cost())
job.upload()
job.start()

data = None
for attempt in range(8):
    try:
        data = job.load(path="artifacts/tidy3d/loss_probe_probe_12.hdf5")
        print(f"downloaded on attempt {attempt+1}")
        break
    except Exception as exc:
        print(f"download attempt {attempt+1} failed: {exc}")
        time.sleep(10)

if data is None:
    raise RuntimeError("could not download SimulationData for probe 12b")

m_out = data["out_0"]
m_refl = data["refl_0"]
fwd = m_out.amps.sel(direction="+", mode_index=0)
bwd = m_out.amps.sel(direction="-", mode_index=0)
r_fwd = m_refl.amps.sel(direction="+", mode_index=0)
r_bwd = m_refl.amps.sel(direction="-", mode_index=0)
fwd_pow = np.abs(fwd.values)**2
bwd_pow = np.abs(bwd.values)**2
r_fwd_pow = np.abs(r_fwd.values)**2
r_bwd_pow = np.abs(r_bwd.values)**2
print("out_0 f(THz) =", [float(fr/1e12) for fr in m_out.amps.coords['f'].values])
print("out_0 |a_fwd|^2 =", [float(v) for v in fwd_pow], "mean =", float(np.mean(fwd_pow)))
print("out_0 |a_bwd|^2 =", [float(v) for v in bwd_pow], "mean =", float(np.mean(bwd_pow)))
print("refl_0 |a_fwd|^2 =", [float(v) for v in r_fwd_pow], "mean =", float(np.mean(r_fwd_pow)))
print("refl_0 |a_bwd|^2 =", [float(v) for v in r_bwd_pow], "mean =", float(np.mean(r_bwd_pow)))
nc = np.asarray(m_out.n_complex.values)
print("out_0 n_complex.real =", [float(v.real) for v in nc.flatten()])
omega = 2*np.pi*np.asarray(m_out.amps.coords['f'].values)
phase = np.unwrap(np.angle(fwd.values))
slope = float(np.polyfit(omega - omega.mean(), phase.flatten(), 1)[0])
ng = 299792458.0 * slope / (20.0e-6)
print("fitted n_g (20 um) =", round(ng,4))
print("DONE")
