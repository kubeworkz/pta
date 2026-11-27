"""Submit the multi-line adjoint, then carry its density through import-em and the four runs.

The comb cannot be one inverse-design job: this ``tidy3d`` release rejects a metric with more
than one frequency, and a source-free design simulation is rejected at upload. The problem is
therefore built as ``InverseDesignMulti`` -- one simulation per (comb line, input port), each
with its own mode source -- and the objective is a differentiable ``post_process_fn`` over the
resulting ``BatchData``.

Usage::

    PTA_STEPS=1 ./.venv/Scripts/python.exe run_design_submit_v8.py
    PTA_ESTIMATE_ONLY=1 ./.venv/Scripts/python.exe run_design_submit_v8.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np

CONFIG = Path.home() / ".config" / "tidy3d" / "config.toml"
os.environ["TIDY3D_API_KEY"] = tomllib.loads(CONFIG.read_text(encoding="utf-8"))["web"]["apikey"]

import tidy3d as td  # noqa: E402
import tidy3d.plugins.invdes as tdi  # noqa: E402
import tidy3d.web as web  # noqa: E402

from pta import blocks, pdk, polygons, t3d  # noqa: E402
from pta import layout as layout_module  # noqa: E402
from pta.config import load_config  # noqa: E402
from pta.core import build_core  # noqa: E402

cfg = load_config()
tidy3d_dir = Path(cfg.artifacts_dir) / "tidy3d"
tidy3d_dir.mkdir(parents=True, exist_ok=True)
pdk.install(cfg)
history_path = tidy3d_dir / "design_v8_history.hdf5"

core = build_core(cfg)
design, objective = t3d.build_inverse_design(cfg, core.weights)
t3d.validate_design(design)
steps = int(os.environ.get("PTA_STEPS", "1"))

print(
    f"design: {len(design.simulations)} simulations "
    f"({cfg.K} comb lines x {cfg.N} input ports), {steps} optimizer step(s)",
    flush=True,
)
print("tasks:", ", ".join(design.task_names), flush=True)

if os.environ.get("PTA_ESTIMATE_ONLY") == "1":
    probe = web.Job(simulation=design.simulations[0], task_name="pta_core_cost_probe")
    probe.upload()
    per_job = probe.estimate_cost()
    print(f"estimate: {per_job} credits per simulation")
    print(f"          {per_job * len(design.simulations) * steps} credits for {steps} step(s)")
    raise SystemExit(0)

start_step = int(os.environ.get("PTA_START_STEP", "1"))
if os.environ.get("PTA_RESUME") == "1":
    result = tdi.InverseDesignResult.from_file(str(history_path))
    print(
        f"resumed from {history_path}: {len(result.objective_fn_val)} step(s) done, "
        f"continuing at scheduled step {start_step}",
        flush=True,
    )
else:
    result = None

# The spec's beta schedule is one beta per epoch, so each step rebuilds the design region
# at its own projection steepness rather than running one fixed beta.
started = time.time()
for offset in range(steps):
    beta = t3d.design_beta(start_step + offset)
    design, objective = t3d.build_inverse_design(cfg, core.weights, beta=beta)
    step_optimizer = tdi.AdamOptimizer(
        design=design,
        learning_rate=0.05,
        num_steps=1,
        results_cache_fname=str(history_path),
    )
    if result is None:
        params0 = t3d.design_initial_density(cfg).astype(float)[:, :, None]
        state = step_optimizer.initial_state(params0)
        result = tdi.InverseDesignResult(
            design=design, opt_state=(state,), params=(params0,)
        )
    result = step_optimizer.continue_run(result=result, num_steps=1, post_process_fn=objective)
    print(
        f"step {start_step + offset} beta={beta:.3f} "
        f"objective={float(result.objective_fn_val[-1]):.4f} "
        f"penalty={float(result.penalty[-1]):.4f}",
        flush=True,
    )
print(f"adjoint finished in {(time.time() - started) / 60:.1f} min", flush=True)
print("objective history:", [float(value) for value in result.objective_fn_val], flush=True)
print("penalty history:  ", [float(value) for value in result.penalty], flush=True)

density = t3d.design_density(cfg, result, design)
density_path = tidy3d_dir / "design_density.npz"
np.savez_compressed(density_path, density=density)
print(f"density: {density_path} {density.shape} [{density.min():.3f}, {density.max():.3f}]")

design_data = t3d.write_design_data(cfg, design, result, tidy3d_dir / "design_em.hdf5")
print(f"design data: {design_data}")

core_em = build_core(cfg, density=density, backend="tidy3d")
polygons_um = polygons.density_to_polygons(density, cfg)
component, landings = blocks.freeform_core(cfg, core_em, polygons_um, cfg.freqs_hz)
gds_path = layout_module.export_gds(component, tidy3d_dir / "core_adjoint.gds")
print(f"core GDS: {gds_path} ({len(polygons_um)} polygons)")
print("stub landings (um):", {name: round(value, 3) for name, value in landings.items()})

verify_paths = []
for index in range(cfg.N):
    simulation = t3d.build_verification_sim(cfg, gds_path, index)
    path = tidy3d_dir / f"verify_adjoint_in{index}.hdf5"
    web.run(
        simulation,
        task_name=f"pta_verify_adjoint_in{index}",
        path=str(path),
        verbose=True,
    )
    verify_paths.append(path)
    print(f"verify_in{index}: {path}")

core_out = Path(cfg.artifacts_dir) / "core_smatrix_adjoint.npz"
command = [
    sys.executable,
    "-m",
    "pta",
    "import-em",
    "--design-data",
    str(design_data),
    "--verify-data",
    *[str(path) for path in verify_paths],
    "--core-out",
    str(core_out),
]
env = dict(os.environ, PYTHONPATH="src", PYTHONIOENCODING="utf-8")
print("$", " ".join(command))
imported = subprocess.run(command, env=env, text=True)
print(imported.stdout, end="")
if imported.returncode != 0:
    raise SystemExit(f"import-em failed with exit {imported.returncode}")

run_dir = Path(cfg.artifacts_dir) / "run_core_adjoint"
command = [
    sys.executable,
    "-m",
    "pta",
    "all",
    "--core",
    str(core_out),
    "--out",
    str(run_dir),
]
print("$", " ".join(command))
final = subprocess.run(command, env=env, text=True)
print(final.stdout, end="")
raise SystemExit(final.returncode)
