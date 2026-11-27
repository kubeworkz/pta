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
| `tidy3d_design_spec.json` | Tidy3D | Adjoint problem: stack, region, grid, targets, objective |
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
  t3d.py                Tidy3D design spec and verification simulators
  bundle.py, io.py      portal bundle, npz interchange
  t3d.py, em_import.py  Tidy3D specs and solved-core import
  pipeline.py, cli.py   orchestration and command line
tests/                  config, physics, DRC, pipeline regression tests
```

## Limitations

* The freeform core is an analytic surrogate until the Tidy3D adjoint + verification run.
* DRC is morphological, not the foundry deck; re-run the PDK rules before tapeout.
* The LO comb is delivered off chip to each hybrid (24 ports); the on-chip 1×N splitter is
  delivered separately because its fan-out cannot be routed crossing-free in this floorplan.
* Driver/TIA ASICs, fibre attach and package are out of scope for v0.1.

See [`docs/implementation-plan.md`](docs/implementation-plan.md) for phases, gates, the
decision log and the risk register.
