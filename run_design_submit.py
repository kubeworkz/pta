import sys, time, os
sys.path.insert(0, "src")
from pathlib import Path

# Re-inject the Tidy3D key from the user's local config into the process env,
# matching the earlier submission path. Do not print the key.
cfg_path = Path.home() / ".config" / "tidy3d" / "config.toml"
import tomllib
data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
os.environ["TIDY3D_API_KEY"] = data["web"]["apikey"]

import tidy3d as td
import tidy3d.web as web
from pta.config import load_config

cfg = load_config()
spec_path = cfg.path("t3d_design")
sim = td.Simulation.from_file(str(spec_path))
print("loaded design spec from", spec_path)
print("sim type:", type(sim))

job = web.Job(simulation=sim, task_name="pta_design_P6")
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
        names = [s.name for s in sl.structures]
        print("sim_last.structures names:", names)
        for s in sl.structures:
            m = getattr(s, "medium", None)
            print(
                "  structure", getattr(s, "name", None),
                "medium type:", type(m),
                "is CustomMedium-ish:",
                isinstance(m, (td.CustomMedium, td.CustomAnisotropicMedium)),
            )
            eps_bounds = getattr(getattr(m, "eps_bounds", None), "__iter__", lambda: ())()
            try:
                low, high = next(iter(eps_bounds)), next(iter(eps_bounds))
            except Exception:
                low = high = None
            print("    eps_bounds:", getattr(m, "eps_bounds", None))

print("DONE")
