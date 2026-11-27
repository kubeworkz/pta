# Implementation plan — PTA golden build

Coherent frequency-comb photonic tensor accelerator, built with PhotonForge (layout,
circuit model, DRC, GDSII) and Tidy3D (electromagnetic verification and inverse design).
Source design: [`pta-tiny3d.md`](pta-tiny3d.md).

Status: **v0.1 golden build generated and self-consistent**; the freeform core is still an
analytic surrogate pending its Tidy3D solve (gate G3 below).

---

## 1. Objective and scope

Deliver a tapeout-ready GDSII of an N×N coherent photonic tensor core that multiplies a
signed input vector by a signed weight matrix on K WDM comb lines, plus the models and
reports needed to defend the numbers:

| Deliverable | Definition of done |
|---|---|
| Layout | DRC-clean GDSII of the chip and of the standalone freeform core |
| Kernel | Signed complex 4×4 matrix per comb line, passive and reciprocal |
| Verification | Circuit-level MVM error < 2 % of full scale, per line |
| System model | TOPS, wall-plug power, TOPS/W, SNR and ENOB from the same config that built the layout |
| Portal hand-off | Tidy3D job specs + PhotonForge bundle, no manual editing |

Out of scope for v0.1: driver/transimpedance ASIC design, fibre attach, package and thermal
mechanical design, and the foundry's proprietary DRC deck.

---

## 2. Architecture as built

```
COMB_SIG_IN_i,k ──► TFLN MZM ──► ring WDM mux ──► freeform core ──► 180° hybrid ──► balanced PD
   (12 fibres)      (V=Vπ bias)     (3 rings)       (4×4, 18×20 µm)     ▲
                                                                        │
                                              COMB_LO_IN_i (off-chip LO comb)
```

* **Comb**: K = 3 lines at 1545 / 1550 / 1555 nm (625 GHz spacing), 4 mW per line.
* **Core**: 4 inputs × 4 outputs, 18 × 20 µm freeform region, 140 nm feature limit,
  0.1 dB excess loss, one complex S-matrix per line.
* **Signed weights**: null-biased push-pull MZMs set the magnitude, the 180° hybrid plus
  balanced detector difference gives the sign — no negative optical amplitudes are needed.
* **Lane phases**: one thermo-optic trimmer per lane, calibrated with a pilot tone.
* **Chip**: 11.819 × 2.195 mm, 18 blocks, 24 optical ports, 20 routed connections.

Golden-build results (`configs/pta.yaml`, seed 7):

| Metric | Value |
|---|---|
| Signed MVM error (worst line) | **0.51 %** (0.07 / 0.11 / 0.51 %) |
| Throughput | 2.4 TOPS at 25 GBd |
| Wall-plug power | 2.60 W (laser 1.20, RF 0.74, DAC/ADC 0.60, thermal 0.07) |
| Energy efficiency | 0.92 TOPS/W, 2167 fJ/MAC |
| Receiver | 48.0 dB SNR, 7.68-bit ENOB |
| DRC | core 140 nm **pass**, chip 800/250 nm **pass** |

---

## 3. Decisions and deviations from the design document

| # | Design doc says | Built as | Why |
|---|---|---|---|
| D1 | `pf.parametric.route`, `check_min_width`, `PortSpec` helpers | PhotonForge 2.2 public API (`Component`, `Path`, `SMatrix`, `CircuitModel`, `Technology`) | Those helpers do not exist in 2.2; the layout is written against the installed API and introspected signatures |
| D2 | Tidy3D adjoint plugin drives the core | `run_design_submit_v8.py` runs the adjoint locally as an `InverseDesignMulti`; `tidy3d_design_spec.json` still records the problem statement, and 4 forward verification simulators close the loop | `tidy3d.plugins.adjoint` is gone in Tidy3D 2.x, and `tidy3d.plugins.invdes` takes one frequency per metric, so the comb is one simulation per line and input port |
| D3 | On-chip 1×N LO splitter feeding every hybrid | Standalone 1×N MMI star coupler + GDS; LO enters each hybrid from off chip | A crossing-free on-chip fan to the hybrid column is impossible in this floorplan; the splitter is still delivered as a characterised component |
| D4 | Fixed lane phases | Per-lane thermo-optic trimmer + pilot-tone calibration | Arbitrary lane phases alone produced 375 % MVM error; calibration brings it to 0.5 % |
| D5 | Single DRC rule set | Two rule sets (`freeform` 140 nm on the core, `waveguide` 800/250 nm on the chip) with the core excluded via a device marker on layer (99,0) | The 140 nm freeform features are legal only inside the inverse-designed region |
| D6 | Core S-matrix from FDTD | Analytic surrogate, labelled `backend="analytic"`, replaced by the Tidy3D S-matrix via `--core` | Keeps the whole flow runnable without cloud credits; the ingestion path is already implemented and tested |

---

## 4. Work breakdown

Phases are sequential, each with a binary exit gate. "Done" means the gate passes in this
repository today.

### P0 — Foundations · **done**
PDK (`Technology`, layers, `"SiN"` port spec), frozen `AcceleratorConfig` with YAML
round-trip and CLI overrides, artifact naming, CLI skeleton, `.venv` with
`photonforge 2.2.0`, `tidy3d 2.12.2`, `gdstk 1.0.2`.
**Gate G0**: `pta design` runs from a clean checkout; config round-trips byte-identically.

### P1 — Kernel and EM hand-off · **done (surrogate)**
Target matrices with unit spectral norm, complex transmission with modelled fabrication
phase error, reciprocal/passive 2N×2N S-matrix, density map and polygons, Tidy3D design
spec and per-input verification simulators.
**Gate G1**: reciprocity error < 1e-9, σ_max ≤ 1 on every line; every Tidy3D JSON validates
as a `td.Simulation`.

### P2 — Block library · **done (analytic models)**
TFLN MZM (1×2 MMI splitter, 5 mm push-pull arms, GSG electrodes, Vπ = 5 V), ring WDM mux
(R = 12 µm, Q = 20 k, 5 nm grid), 180° hybrid (2×2 MMI + 90° section), 1×N MMI splitter,
Euler S-bend connector with heater trimmer.
**Gate G2**: every block exposes ports at the documented pitch and returns an S-matrix on
the comb grid.

### P3 — Assembly, routing, DRC · **done**
Left-to-right floorplan, 20 deterministic Euler-bend routes, no crossings, device marker,
GDSII export for core/chip/splitter, exact width/gap/spacing DRC, an opening test for the
freeform core, plus shorts and connectivity.
**Gate G3**: core and chip DRC pass; no overlapping unrelated waveguides; die fits a
single reticle field (11.8 × 2.2 mm ≪ 26 × 33 mm).

### P4 — Verification and system model · **done**
Full-chip PhotonForge circuit solve, homodyne field reconstruction, lane-phase calibration,
pilot LO lock, signed MVM comparison, power/SNR/ENOB model, Pareto sweep.
**Gate G4**: worst-line MVM error < 2 %; ENOB in (1, 16) bits; power breakdown closes.

### P5 — Portal packaging · **done**
`dist/pta_portal_bundle.zip` with GDSII, S-matrices, manifests, DRC/verification/metrics
reports, Tidy3D specs and `README_UPLOAD.md`.
**Gate G5**: bundle unpacks standalone; every JSON parses; `portal_manifest.json` lists
each file's purpose and portal.

### P6 — Electromagnetic sign-off · **next**
First run `tidy3d_validate_straight.json` and follow
[`tidy3d-validation.md`](tidy3d-validation.md): a straight guide must show a forward
amplitude near unity, a weak backward direction and a linear phase slope. Only then trust
the full-core extraction.
1. Run `run_design_submit_v8.py`. It builds the `InverseDesignMulti` (one simulation per
   comb line and input port, each with its own mode source), optimises the mask against the
   phase/gain-invariant overlap under the conic filter and beta schedule with the
   erosion/dilation penalty, and writes the density, the design `SimulationData` and the
   core GDS. `PTA_RESUME=1 PTA_START_STEP=<n>` continues the schedule from the history.
2. Re-run the 4 `tidy3d_verify_in*.json` simulations, export each input's `(K, N)`
   output amplitudes, and assemble `core_smatrix.npz` with
   `pta import-em --density <mask.npy> --monitors <dir>` (validates the density grid,
   enforces reciprocity/passivity and attaches the mask). With the downloaded
   `SimulationData` files, one command does the extraction too:
   `pta import-em --design-data <design.hdf5> --verify-data <verify_in0.hdf5> ...`,
   or call `pta.em_import.import_from_simulation_data` from Python. The density reader
   accepts `InverseDesignResult.sim_last`, isotropic and anisotropic custom media, and the
   design region's `eps_bounds`; an explicit density array is the fallback.
3. `pta all --core artifacts/core_smatrix.npz` — the layout, DRC, verification and metrics
   regenerate against the EM kernel with no code changes.
4. Re-issue the bundle with `backend="tidy3d"`.
**Gate G6**: EM-extracted MVM error < 2 %; the EM and surrogate kernels agree within the
fabrication tolerance budget (60 mrad phase, 0.2 dB amplitude).

---

## 5. Verification gates in detail

| Gate | Check | Threshold | Where |
|---|---|---|---|
| Physics | `|S − Sᵀ|`, σ_max(S) | < 1e-9, ≤ 1 | `core.physics_report` |
| Layout | width / gap / spacing / opening defects, shorts, connectivity failures | 0 | `drc.check_gds` |
| Circuit | signed MVM, worst line | < 2 % | `verify.verify_mvm` |
| Receiver | SNR, ENOB | > 20 dB, > 4 bits | `metrics.evaluate` |
| Regression | pytest suite (config, physics, DRC, pipeline) | all pass | `tests/` |

The DRC is exact, not tolerance-based. Width is measured per drawn primitive by solving the
rectangle relation `w² − (P/2)w + A = 0` for its small root, which is exact for a strip of
any length and aspect ratio, so a 0.8 µm band reads 0.7998 µm wherever it sits. The gap rule
dilates the two pieces of one cell until they merge and reports the merge distance. The
golden build measures a narrowest band of 0.7998 µm against the 0.800 µm rule (−0.02 %) and
a smallest gap of 0.2988 µm against 0.250 µm; the 4 nm resolution is 0.5 % of the width and
covers the polygon sampler's own 1.5–2.2 nm error plus the 1 nm boolean grid.

Three structural checks back the measurements up. The **short** check reports any
positive-area overlap between different cell references. The **cross-reference spacing**
check measures the actual clearance between the material of different cells, which the
per-cell gap rule cannot see: two routes drawn in separate references can miss each other
by 0.1 µm and each cell still passes on its own. A butt joint (distance zero along a
shared edge) is accepted only when `layout_manifest.json` declares the route between those
two references **and** all of the contact lies within half the joint width of the port
recorded for it. One predicate covers every wrong shape: an undeclared edge-to-edge
contact fails as an undeclared contact, and a declared pair whose contact reaches past the
joint cap fails as contact outside the port — whether the joint is in the wrong place, a
second touch exists elsewhere, or the route keeps running along the block edge. The golden
build has 56 near pairs, 40 declared joints at their ports — the 32 routed ones and the 8
core-stub ones, since the freeform cell is excluded from the 800 nm width and gap rules but
not from spacing — 0 undeclared, 0 outside the port, a widest accepted band of 0.404 µm
against the 0.412 µm limit, and a smallest separated clearance of 3.2 µm. The **connectivity** check counts merged
components per cell (mux 2K+1, hybrid 2, everything else 1), which is the only rule that
sees two nets fused into one piece of material: a fusion has no thin geometry and no gap to
measure, so neither a width nor a gap rule can report it. The earlier 0.35 µm notch
allowance let a mux through whose ring overlapped its bus by 0.1 µm; the component count
does not (1 instead of 7).

The core uses erosion for its mean width plus an **opening** test at `w/2`, because
erosion only reports a component that vanishes or splits: a sub-rule spur disappearing from
the edge of a surviving island leaves the component count untouched. The opening residual
is the material that no inscribed disk of radius `w/2` covers, which is the morphological
definition of a feature narrower than `w`. Convex corners fail the same test by geometry, so
each residual is compared with the cap reach `r / tan(θ/2)` of the corner it touches; only
material beyond every such cap is reported. The generated core measures 0.252 µm width,
0.281 µm gap and zero opening defects against 0.140 µm. The eight stubs are **access
waveguides**: each runs from its chip port into the design region until it shares at least
0.1 µm² with an island, which takes 0.2–3.2 µm depending on the port; those leads are not
part of the optimized scatterer, their depth per port and the applied limit are recorded in
`drc_core.json`, and a port that finds no island within that limit — 6 µm for the surrogate,
0.6 µm once a Tidy3D core is ingested — fails the build. This is still a design aid, not the
foundry deck — sign-off must re-run the PDK's own rules and LVS on `pta_chip.gds`.

---

## 6. Portal hand-off

**Tidy3D** (`artifacts/`, also inside the bundle):

| File | Use |
|---|---|
| `tidy3d_design_spec.json` | Adjoint problem: stack, design region, grid, targets, conic filter radius, tanh projection and beta schedule, erosion/dilation penalty, phase-invariant objective |
| `tidy3d_verify_in0..3.json` | Forward 3D FDTD of the exported core, one per input port; returns the S-matrix column |
| `tidy3d_validate_straight.json` | Minimal straight waveguide on lane 0; validates monitor direction, absolute normalization and phase before the core runs |

Both are complete `td.Simulation` JSON documents built and validated locally — no API key
or credits are consumed by this repository. Upload them in the Tidy3D web UI (or
`td.web.upload`), then download the S-matrix and feed it back with `--core`.

**PhotonForge**: upload `pta_chip.gds` (full chip), `pta_core.gds` (core only, for the
inverse-design loop) and `lo_splitter.gds`. The circuit model is reproducible from
`chip_smatrix.npz` + `layout_manifest.json` without re-solving.

---

## 7. Risk register

| Risk | Impact | Mitigation / status |
|---|---|---|
| Core is an analytic surrogate | Numbers are optimistic until EM sign-off | G6 is a hard gate; ingestion path already wired (`--core`) |
| 5 mm MZM, Vπ = 5 V at 25 GBd | Driver swing and RF loss | 50 Ω GSG, 0.735 W RF budget; driver is a v0.2 item |
| Ring thermal drift (Q = 20 k, 5 nm grid) | Channel crosstalk | 2.5 mW heater per ring, 25 heaters, 0.07 W total |
| Laser wall-plug dominates (46 % of power) | Caps TOPS/W at ~0.9 | Higher-WPE laser or fewer lines; the Pareto sweep quantifies it |
| Off-chip LO per hybrid (24 ports) | Fibre attach cost | Accept for v0.1; on-chip splitter ready but unroutable in this floorplan (D3) |
| Morphological DRC ≠ foundry deck | False sense of cleanliness | Explicit limitation; run the PDK deck before tapeout |
| Freeform polygon count/size | Mask write time, data volume | Density map is simplified to ≥ 140 nm features; core GDS is 46 kB |

---

## 8. Module map

| Design-doc section | Module | Responsibility |
|---|---|---|
| §2 architecture, §4 comb | `config.py` | Frozen config, wavelengths, port pitch, layer map |
| §5 freeform core | `core.py`, `polygons.py` | S-matrix, density map, contour extraction, fidelity |
| §6 MZM / rings / hybrid | `blocks.py`, `smodels.py` | Block geometry and per-block S-matrices |
| §7 floorplan | `layout.py` | Placement, Euler-bend routing, manifests, GDSII |
| §8 DRC | `drc.py` | Exact width/gap/spacing/opening measurement, shorts, connectivity, reports |
| §9 verification | `verify.py` | Homodyne fields, lane calibration, signed MVM |
| §10 link budget | `metrics.py` | TOPS, power, SNR, ENOB, Pareto |
| §11 Tidy3D | `t3d.py`, `em_import.py` | Design spec, forward verification simulators, solved-core import |
| §12 packaging | `bundle.py`, `io.py` | Bundle, npz interchange |
| — | `pipeline.py`, `cli.py` | Orchestration and command line |

---

## 9. Next revision (v0.2)

1. Run P6 and replace the surrogate with the EM kernel.
2. Add the driver/ADC power model at the actual 25 GBd signal integrity level.
3. Route the on-chip LO splitter (needs either a second metal layer for crossings or a
   hybrid column with ports facing the splitter).
4. Add thermal crosstalk between neighbouring rings to the circuit model.
5. Generate the foundry DRC/LVS deck run and a mask-layout review checklist.
