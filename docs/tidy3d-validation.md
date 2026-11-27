# Tidy3D direction, normalization and phase check

`artifacts/tidy3d/tidy3d_validate_straight.json` is a minimal straight SiN waveguide used
to validate the assumptions behind the Tidy3D importer before trusting a full-core result.
It shares the source and monitor builders with `tidy3d_verify_in*.json`; the intended
differences are the straight geometry and the fact that only lane 0 is populated.

## What is simulated

- A 21 µm SiN strip (`design_lx + 2*stub_len`) on the SiO2 cladding, with the TFLN slab
  above it. The guide ends at ±10.5 µm, matching the core access stubs.
- Mode source at `x = -design_lx/2 - 1 = -10 µm`, `y = port_y(0)`, forward `"+"`. The guide
  starts 0.5 µm before the source plane, so the source launches the guided SiN mode.
- `out_0` mode monitor at `x = +10 µm`. It is 20 µm from the source and 0.5 µm before the
  output facet, mirroring the monitor position inside the core output stub.
- `refl_0` mode monitor 0.4 µm before the source.
- The same source size and pulse, monitor size, mode spec, comb frequencies and PML
  boundaries as the core verification specs. The simulation is centred on lane 0, so the
  guide sits in the middle of the domain.

The group-index fit needs the **source-to-output-monitor distance**, 20 µm, not the 21 µm
physical guide length. The checker reads both centres from the `SimulationData`, so there is
no length to pass or get wrong.

## Run it end to end

From the project root, configure the Tidy3D API key once (or set `TIDY3D_API_KEY`):

```bat
.venv\Scripts\tidy3d.exe configure
```

With the venv active, submit the spec and download the result:

```python
import tidy3d as td
import tidy3d.web as web

sim = td.Simulation.from_file("artifacts/tidy3d/tidy3d_validate_straight.json")
web.run(
    sim,
    task_name="pta_straight_validation",
    folder_name="pta",
    path="artifacts/tidy3d/tidy3d_validate_straight.hdf5",
)
```

`web.run` prints the task URL and estimated FlexCredit cost, then writes the HDF5. Run the
checker:

```bash
python -m pta.validate_straight artifacts/tidy3d/tidy3d_validate_straight.hdf5
```

It prints the forward and backward powers at each comb line, the reflection-monitor power
and the group index fitted over the derived source-to-monitor distance.

## Checklist

1. **Load the result.** `td.SimulationData.from_file(...)` succeeds and exposes `out_0` and
   `refl_0` with `direction`, `f` and `mode_index` dimensions.
2. **Forward direction.** For `out_0`, `amps.sel(direction="+", mode_index=0)` is the strong
   signal; `direction="-"` is at least 10 dB weaker. The uncoated output facet alone reflects
   roughly 1–2% (about −18 dB), so do not require −20 dB. If `"+"` is dark and `"-"` carries
   the power, the importer's direction convention is wrong — stop and fix it before importing
   a core.
3. **Absolute normalization.** The forward power `|a|²` is close to 1. Over the 20 µm path
   the propagation loss is under 0.001 dB, so expect roughly 0.95–1.0 (the exact value
   depends on the mode overlap at source and monitor and any facet reflection). A value
   near 0.25 or 2 means the portal data needs a source-amplitude or power normalization
   before it can be used as an S-matrix.
4. **Reflection monitor.** For `refl_0`, the backward `direction="-"` amplitude is small
   compared with forward — typically around −18 dB from the input facet. Below −10 dB
   confirms which direction label corresponds to back-reflection; a value comparable to
   forward means the monitor or direction is wrong.
5. **Phase.** Unwrap `angle(out_0.amps)` along the comb and fit it against angular frequency.
   The slope should be linear and give a sensible group index via
   `n_g = +c * dφ/dω / L`, where the script takes `L = 20 µm` from the source and monitor
   centres. The sign follows Tidy3D's `e^{-iωt}` convention, in which a forward wave has
   `exp(+ikx)` and its phase increases with frequency. The absolute phase is not meaningful
   (source phase and monitor reference are arbitrary); the slope and the relative phase
   across the three comb lines are.
6. **Record the numbers.** Keep the forward/backward powers, reflection level and fitted
   group index with the result, then repeat the same checks on a full-core verify result.

The script selects `direction="+"` and mode 0 for the forward output, `direction="-"` for
the backward output, and `direction="-"` on `refl_0` for the reflection. It centres the
frequency axis before fitting so the ~1e15 Hz values do not lose precision.

If steps 2–3 fail, do not import a full-core result: the reciprocal reconstruction would be
built on a wrong direction or scale. If they pass, the same `out_*` extraction can be used
for the four core verification runs.
