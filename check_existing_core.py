import sys, json
sys.path.insert(0, "src")
from pta.config import load_config
from pta.core import CoreModel, physics_report, fidelity
from pta.validate_straight import evaluate, format_report
import numpy as np
import tidy3d as td

cfg = load_config()
print("config: N=", cfg.N, "K=", cfg.K, "alpha_scale=", cfg.alpha_scale, "drc_freeform_um=", cfg.drc_freeform_um)
core = CoreModel.from_npz("artifacts/core_smatrix.npz")
print("loaded core: backend=", core.backend, "N=", core.N, "K=", core.K, "alpha=", core.alpha)
print("density present:", core.density is not None, "shape:", (core.density.shape if core.density is not None else None))
print("s_matrix shape:", core.s_matrix.shape)
print()
print("physics_report:")
print(json.dumps(physics_report(core), indent=2))
print()
print("fidelity:")
print(json.dumps(fidelity(core), indent=2))
print()
data = td.SimulationData.from_file("artifacts/tidy3d/tidy3d_validate_straight.hdf5", monitor_names=["out_0", "refl_0"])
report = evaluate(data)
print("straight-waveguide normalization reference:")
print(format_report(report))
print()
print("forward normalization reference =", round(float(np.mean(report.forward_power)), 4))
