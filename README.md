# PTA — coherent frequency-comb photonic tensor accelerator

Golden-build implementation of [`docs/pta-tiny3d.md`](docs/pta-tiny3d.md): a 4×4 signed
photonic matrix multiplier on three WDM comb lines, laid out with **PhotonForge 2.2** and
handed off to **Tidy3D** for electromagnetic verification and inverse design.

The whole flow runs locally: `pta all` designs the core, assembles the chip, routes it,
checks DRC, solves the circuit, verifies the signed MVM and writes the portal bundle.

```
COMB_SIG_IN_i,k ──► TFLN MZM ──► ring WDM mux ──► freeform core ──► 180° hybrid ──► balanced PD
   (12 fibres)      (V=Vπ bias)     (3 rings)       (4×4, 18×20 µm)     ▲
                                                                        │
                                              COMB_LO_IN_i (off-chip LO comb)
```

## Golden-build results

| Metric | Value |
|---|---|
| Chip | 11.819 × 2.195 mm, 18 blocks, 24 ports, 20 routes |
| Kernel | 4×4 signed complex, K = 3 lines @ 1545/1550/1555 nm |
| Signed MVM error | **0.51 %** worst line (0.07 / 0.11 / 0.51 %) |
| Throughput | 2.4 TOPS at 25 GBd |
| Power / efficiency | 2.60 W, 0.92 TOPS/W, 2167 fJ/MAC |
| Receiver | 48.0 dB SNR, 7.68-bit ENOB |
| DRC | core 140 nm pass; chip 800/250 nm pass — 0.7998 µm width, 0.2988 µm gap measured |

## Install

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Requires Python ≥ 3.10 and `photonforge`, `tidy3d`, `gdstk`, `numpy`, `scipy`, `pyyaml`.

## Quick start

```bash
pta all                      # design → layout → DRC → verify → metrics → bundle
pta all --no-drc             # skip the polygon DRC (faster iteration)
pta design --set N=8 --set wavelengths_um=[1.54,1.55,1.56,1.57]
pta layout && pta verify && pta metrics
pta bundle                   # repackage existing artifacts
pytest                       # regression suite
```

Options are accepted before or after the subcommand: `--config FILE`, `--set KEY=VALUE`
(repeatable), `--out DIR`, `--inputs FILE.npz`, `--core FILE.npz`, `--seed N`, `--no-drc`.

Point the pipeline at an electromagnetic core with `pta all --core artifacts/core_smatrix.npz`;
the schema is the one `pta design` writes, so the Tidy3D S-matrix drops in without code
changes. Build that file from the solved jobs with:

```bash
pta import-em --density optimized_density.npy --monitors tidy3d_results/
pta all --core artifacts/core_smatrix.npz
```

The monitor directory holds `verify_in0..N-1.npz`, one per input port, with a complex
`amplitudes` array of shape `(K, N)` (`amplitudes[k, o]` = transmission to `OUT_o` at line
`k`) and optionally `freqs_hz`. The density is a `(cfg.nx, cfg.ny)` map in `[0, 1]`, saved
as `.npy` or as `.npz` with a `density` key. `import-em` checks the grid, assembles the
reciprocal S-matrix, refuses a non-passive result and writes `core_smatrix.npz` with the
density attached. An ingested core is held to the 0.6 µm stub-landing limit rather than the
surrogate's 6 µm, so its ports must reach the access waveguides.

When the solved `SimulationData` files are available, extract and import them directly — no
manual monitor export:

```bash
pta import-em --design-data design.hdf5 \
  --verify-data verify_in0.hdf5 verify_in1.hdf5 verify_in2.hdf5 verify_in3.hdf5
```

The command reads the four verification files' `out_0..N-1` mode monitors (`direction="+"`,
mode 0) and maps the design result's optimised custom medium back to a density. Both
`CustomMedium` and `CustomAnisotropicMedium` are accepted (the `xx` component by default),
as are `InverseDesignResult.sim_last`, `SimulationData` and a raw `Simulation`. The mapping
uses the result's `design_region.eps_bounds`, falling back to the SiO2/SiN values; pass
`--structure-name` if the design data contains more than one custom medium. From Python,
`extract_density` also accepts an explicit density array or an `eps_bounds=` override, and
`import_from_simulation_data` performs the whole import.

## Tidy3D inverse-design chain

`run_design_submit_v8.py` is the runner: it builds and validates the adjoint problem,
optimises the freeform mask, exports the mask as GDS and `SimulationData`, re-runs the four
verification simulations, then calls `pta import-em` and `pta all --core`.

The installed `tidy3d` 2.12 accepts only one frequency per inverse-design metric and rejects
a source-free design simulation, so the comb is an `InverseDesignMulti`: **one simulation
per comb line and input port**, each with its own mode source at that line's frequency and
`out_*` monitors sampled at that line only. One optimiser step updates the density every
simulation shares.

The objective is the phase- and gain-invariant overlap `|<g, t>|² / (‖g‖²‖t‖²)` with
`g = α·W[k][:, i]·e^{iφ_ref[k]}`, minus a throughput shortfall. That is the quantity the
receiver can actually see: verification fits a pilot gain and per-lane phases before it
scores the kernel. The design region carries the conic filter
(`radius = drc_min_feature = 0.14 µm`) and tanh projection with the advertised beta schedule
(`2.0·1.08^step`, one beta per step), plus an erosion/dilation penalty at the 0.14 µm
freeform scale so the mask is binarised while it is optimised. Every port is seeded with a
0.4 µm landing so the mask stays connected within the 0.6 µm ingested-core limit.

```bash
PTA_STEPS=4 python run_design_submit_v8.py        # epoch count (default 1)
PTA_ESTIMATE_ONLY=1 python run_design_submit_v8.py   # cost estimate, no solve
PTA_RESUME=1 PTA_START_STEP=4 PTA_STEPS=36 python run_design_submit_v8.py   # continue the schedule
```

**Cost.** One epoch bills about **3.9 FlexCredits** (12 simulations at ~0.32 each, adjoint
solve included) and each verification simulation about **1.4**, so the full 40-epoch
schedule plus the four verification runs is roughly **160 credits**. `PTA_ESTIMATE_ONLY=1`
reports only the forward upload of one task (0.074) and understates an epoch by about 4×;
treat 3.9 as the planning number.

**When the balance runs out**, the API rejects the next task submit with HTTP 400 / code
`4000000009` ("You do not have enough FlexCredits … Balance is X, Y is reserved … expected
to cost Z"). The optimizer aborts and the script exits non-zero. No density, GDS or
verification artifact is written by that attempt — those are produced after the loop — but
the epochs that did finish are in `artifacts/tidy3d/design_v8_history.hdf5`. Top up and
continue with `PTA_RESUME=1 PTA_START_STEP=<completed epochs + 1>`.

Before the full-core solve, upload `tidy3d/tidy3d_validate_straight.json` and work through
[`docs/tidy3d-validation.md`](docs/tidy3d-validation.md): a straight guide must show a
forward amplitude near 1, a weak backward direction and a linear phase slope. Run
`python -m pta.validate_straight <result.hdf5>` for the numeric report.

## Artifacts (`artifacts/`, zipped into `dist/pta_portal_bundle.zip`)

| File | Portal | Contents |
|---|---|---|
| `pta_chip.gds` | PhotonForge | Full chip: SiN core, TFLN slab, heaters, RF pads, device marker |
| `pta_core.gds` | PhotonForge / Tidy3D | Freeform core only, for the inverse-design loop |
| `lo_splitter.gds` | PhotonForge | 1×N MMI star coupler, standalone |
| `core_smatrix.npz` | — | Core S-matrix, density map, target weights, backend label |
| `chip_smatrix.npz` | — | Full-chip circuit S-matrix, `(K, P, P)` complex |
| `layout_manifest.json` | PhotonForge | Bounding boxes, ports, connections, per-block model status |
| `drc_core.json`, `drc_chip.json` | PhotonForge | Width, gap, cross-reference spacing, opening and connectivity defects |
| `verification.json` | — | Per-line ideal vs measured MVM, lane calibration, pass flag |
| `metrics.json` | — | TOPS, power breakdown, SNR, ENOB |
| `tidy3d_design_spec.json` | Tidy3D | Adjoint problem: stack, region, grid, targets, conic filter + beta schedule, penalty, objective |
| `tidy3d/tidy3d_verify_in*.json` | Tidy3D | Forward 3D FDTD of the core, one per input port |
| `tidy3d/tidy3d_validate_straight.json` | Tidy3D | Straight SiN waveguide for the direction/normalization check |
| `portal_manifest.json`, `README_UPLOAD.md` | both | Upload checklist |

## What is verified vs modelled

| Item | Status |
|---|---|
| Chip assembly, routing, port map | **Verified** — PhotonForge circuit solve, 24 ports, no crossings |
| Signed MVM arithmetic | **Verified** — 0.51 % worst error through the full circuit model |
| Reciprocity, passivity | **Verified** — exact by construction, asserted in tests |
| DRC | **Approximate** — exact 140 nm core (width, gap, opening) and 800/250 nm routing checks with declared-joint spacing, not the foundry deck |
| Freeform core S-matrix | **Surrogate** — analytic, reciprocal, passive, 60 mrad phase error; EM solve pending |
| MZM, rings, hybrid, splitter | **Analytic models** — labelled `analytic` in `layout_manifest.json` |
| System metrics | **Model** — laser/RF/conversion/thermal budget, shot-noise-limited receiver |

The core block is the only one labelled `em_verified`; after the Tidy3D run its extracted
S-matrix replaces the surrogate and the label becomes accurate for the fabricated mask.

## Configuration

`configs/pta.yaml` holds the golden-build values; every field of `AcceleratorConfig` can be
overridden with `--set`. Key knobs: `N`, `wavelengths_um`, `dac_bits`, `mod_length_um`,
`v_pi_l` (V·cm), `ring_q`, `ring_radius_um`, `baud_ghz`, `laser_mw_per_line`, `lo_boost_db`,
`alpha_scale`, and the DRC rules (`drc_min_width`, `drc_min_gap`, `drc_freeform_um`).

## Repository layout

```
configs/pta.yaml        golden-build configuration
docs/                   design document + this implementation plan
src/pta/
  config.py             frozen config, YAML round-trip, artifact names
  weights.py            target matrices, quantization, tiling
  core.py               core S-matrix, density map, fidelity, physics report
  polygons.py           density → GDSII polygons
  pdk.py                PhotonForge technology and layers
  blocks.py             MZM, ring mux, hybrid, splitter, S-bend connector
  smodels.py            per-block S-matrices and DataModels
  layout.py             floorplan, routing, chip assembly, GDSII export
  drc.py                width, gap, spacing, opening, shorts, connectivity
  verify.py             homodyne reconstruction, calibration, signed MVM
  metrics.py            TOPS, power, SNR, ENOB, Pareto sweep
  t3d.py                Tidy3D design spec, inverse-design chain, verification sims
  em_import.py          solved-core import from density + monitor data
  bundle.py, io.py      portal bundle, npz interchange
  pipeline.py, cli.py   orchestration and command line
run_design_submit_v8.py Tidy3D adjoint -> import-em -> four verification runs
tests/                  config, physics, DRC, pipeline regression tests
```

## Limitations

* The freeform core is an analytic surrogate until the adjoint converges: the current run has
  3 of the spec's 40 epochs, and the EM-imported kernel still fails the 2 % MVM gate.
  `run_design_submit_v8.py` resumes from `artifacts/tidy3d/design_v8_history.hdf5`.
* DRC is morphological, not the foundry deck; re-run the PDK rules before tapeout.
* The LO comb is delivered off chip to each hybrid (24 ports); the on-chip 1×N splitter is
  delivered separately because its fan-out cannot be routed crossing-free in this floorplan.
* Driver/TIA ASICs, fibre attach and package are out of scope for v0.1.

See [`docs/implementation-plan.md`](docs/implementation-plan.md) for phases, gates, the
decision log and the risk register.
