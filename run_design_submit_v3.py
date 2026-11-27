import sys, time, os, json
sys.path.insert(0, "src")
from pathlib import Path
from pta.config import load_config
from pta.core import build_core

cfg_path = Path.home() / ".config" / "tidy3d" / "config.toml"
import tomllib
toml = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
os.environ["TIDY3D_API_KEY"] = toml["web"]["apikey"]

import numpy as np
import tidy3d as td
import tidy3d.web as web
import tidy3d.plugins.invdes as tdi
from tidy3d.plugins.expressions.metrics import ModePower

cfg = load_config()
core_analytic = build_core(cfg)
weights = core_analytic.weights
phi_ref = cfg.phi_ref
print("cfg.N=", cfg.N, "cfg.K=", cfg.K, "nx=", cfg.nx, "ny=", cfg.ny)
print("weights shape=", weights.shape, "phi_ref shape=", phi_ref.shape)

# One monitor per output port; each input run reads all out_k monitors.
output_monitor_names = tuple(f"out_{k}" for k in range(cfg.N))

design_region = tdi.TopologyDesignRegion(
    size=(cfg.design_lx, cfg.design_ly, cfg.t_sin),
    center=(0.0, 0.0, 0.0),
    eps_bounds=(cfg.n_sio2**2, cfg.n_sin**2),
    pixel_size=cfg.dl_opt,
)

base = td.Simulation(
    size=(cfg.design_lx + 4.0, cfg.design_ly + 4.0, 2.8),
    run_time=500.0 / float(np.mean(cfg.freqs_hz)),
    grid_spec=td.GridSpec.auto(min_steps_per_wvl=25, wavelength=1.55),
    medium=td.Medium(permittivity=cfg.n_sio2**2),
    boundary_spec=td.BoundarySpec.all_sides(td.PML()),
    plot_length_units="um",
    shutoff=1e-5,
)

metrics = []
for k in range(cfg.K):
    for out_k in range(cfg.N):
        metrics.append(
            ModePower(
                monitor_name=f"out_{out_k}",
                freqs=cfg.freqs_hz,
                direction="+",
                mode_index=0,
            )
        )

design = tdi.InverseDesign(
    design_region=design_region,
    task_name="pta_P6",
    simulation=base,
    metric=tdi.Add(*[ModePower(monitor_name=f"out_{out_k}", freqs=cfg.freqs_hz, direction="+", mode_index=0)
                      for out_k in range(cfg.N) for _ in range(cfg.K)]),
    output_monitor_names=output_monitor_names,
)
print("built InverseDesign; output_monitor_names=", output_monitor_names)

initial_density = cfg.surrogate_density()
params = initial_density.astype(float)[:, :, None]
print("initial density shape=", params.shape, "min/max=", float(params.min()), float(params.max()))

result = tdi.InverseDesignResult(design=design, params=(params,))
print("built InverseDesignResult; sim_last type=", type(result.sim_last))

job = web.Job(simulation=result.sim_last, task_name="pta_P6_inverse_design")
print("estimate:", job.estimate_cost())
job.upload()
print("uploaded")
job.start()
print("started")

out_path = cfg.path("tidy3d") / "tidy3d_design_result.hdf5"
downloaded = None
for attempt in range(10):
    try:
        downloaded = job.load(path=str(out_path))
        print(f"downloaded on attempt {attempt+1}")
        break
    except Exception as exc:
        print(f"download attempt {attempt+1} failed: {exc}")
        time.sleep(12)

if downloaded is None:
    raise RuntimeError("could not download tidy3d_design_result.hdf5")

print("loaded design result type:", type(downloaded))
print("has sim_last:", hasattr(downloaded, "sim_last"))
if hasattr(downloaded, "sim_last"):
    sl = downloaded.sim_last
    print("sim_last type:", type(sl))
    if hasattr(sl, "structures"):
        names = [getattr(s, "name", None) for s in sl.structures]
        print("sim_last.structure names:", names)
        for s in sl.structures:
            m = getattr(s, "medium", None)
            if isinstance(m, (td.CustomMedium, td.CustomAnisotropicMedium)):
                print("  custom medium structure:", getattr(s, "name", None), "type:", type(m))

print("DONE")
