import sys, time
sys.path.insert(0, "src")
import tidy3d as td
import tidy3d.web as web
from pta.config import load_config

cfg = load_config()

# Reload the freshly regenerated wide-monitor spec from disk so the submission
# uses the same file the validator will read, not an in-memory construction.
spec_path = "artifacts/tidy3d/tidy3d_validate_straight.json"
sim = td.Simulation.from_file(spec_path)

print("submitting straight validation (wide-monitor spec)")
print("  out_0.size =", next(m for m in sim.monitors if m.name == "out_0").size)
print("  refl_0.size =", next(m for m in sim.monitors if m.name == "refl_0").size)
print("  TFLN.size  =", next(s for s in sim.structures if s.name == "TFLN").geometry.size)

job = web.Job(simulation=sim, task_name="pta_validate_straight_wide_monitor")
print("estimate:", job.estimate_cost())
job.upload()
job.start()

data = None
for attempt in range(8):
    try:
        data = job.load(path="artifacts/tidy3d/tidy3d_validate_straight.hdf5")
        print(f"downloaded on attempt {attempt+1}")
        break
    except Exception as exc:
        print(f"download attempt {attempt+1} failed: {exc}")
        time.sleep(10)

if data is None:
    raise RuntimeError("could not download SimulationData for straight validation")

# Echo the solver warnings so the extrusion note is visible in the run log.
log = getattr(data, "log", None)
if log:
    for line in log.splitlines():
        if line.strip():
            print("SOLVER_LOG:", line)
else:
    print("no solver log field on SimulationData")

import numpy as np
from pta.validate_straight import evaluate, format_report
report = evaluate(data)
print(format_report(report))
print("DONE")
