import sys, time, os
sys.path.insert(0, "src")
from pathlib import Path

cfg_path = Path.home() / ".config" / "tidy3d" / "config.toml"
import tomllib
data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
os.environ["TIDY3D_API_KEY"] = data["web"]["apikey"]

import numpy as np
import tidy3d as td
import tidy3d.web as web
import tidy3d.plugins.invdes as tdi
from pta.config import load_config

cfg = load_config()
region = tdi.TopologyDesignRegion(
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
)
design = tdi.InverseDesign(
    design_region=region,
    task_name="pta_P6",
    simulation=base,
    objectives=[tdi.Objective(mode_spec=td.ModeSpec(num_modes=1))],
)
initial_density = cfg.surrogate_density()
params = initial_density.astype(float)[:, :, None]
result = tdi.InverseDesignResult(design=design, params=(params,))
print("built InverseDesignResult; sim_last:", type(result.sim_last))

job = web.Job(simulation=result.sim_last, task_name="pta_P6_inverse_design")
print("estimate:", job.estimate_cost())
job.upload()
print("uploaded")
job.start()
print("started")

out_path = cfg.path("tidy3d") / "tidy3d_design_result.hdf5"
data = job.load(path=str(out_path))
print("loaded design result; type:", type(data))
print("has sim_last:", hasattr(data, "sim_last"))
if hasattr(data, "sim_last"):
    sl = data.sim_last
    print("sim_last type:", type(sl))
    if hasattr(sl, "structures"):
        names = [getattr(s, "name", None) for s in sl.structures]
        print("sim_last.structure names:", names)

print("DONE")
