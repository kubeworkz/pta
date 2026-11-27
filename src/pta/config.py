"""Single source of truth for stack, DRC, layer map and system parameters.

Everything downstream (design, layout, verification, metrics, bundling) reads an
:class:`AcceleratorConfig`, so a change made here propagates to every artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

C_0 = 299792458.0  # speed of light, m/s

DEFAULT_LAYERS: dict[str, tuple[int, int]] = {
    "sin_core": (1, 0),
    "tfln_slab": (2, 0),
    "heater": (10, 0),
    "rf_metal": (12, 0),
    "oxide_open": (70, 0),
    "exclude": (99, 0),  # device marker: freeform region exempt from waveguide rules
}

#: Well-known artifact file names, so the CLI, tests and bundler never disagree.
ARTIFACTS = {
    "core_npz": "core_smatrix.npz",
    "core_gds": "pta_core.gds",
    "chip_gds": "pta_chip.gds",
    "lo_gds": "lo_splitter.gds",
    "chip_npz": "chip_smatrix.npz",
    "t3d_design": "tidy3d_design_spec.json",
    "layout": "layout_manifest.json",
    "verify": "verification.json",
    "metrics": "metrics.json",
    "metrics_pareto": "metrics_pareto.json",
    "drc_core": "drc_core.json",
    "drc_chip": "drc_chip.json",
    "bundle": "pta_portal_bundle.zip",
}


@dataclass(frozen=True)
class AcceleratorConfig:
    """Immutable build configuration.

    Defaults reproduce the design document's golden build: a 4x4 coherent core driven by
    three comb lines on the 1545/1550/1555 nm grid, hybrid TFLN-on-SiN stack, 800 nm SiN
    strip, 250 nm DRC gap and 40 um minimum bend radius.
    """

    # Matrix and comb dimensions
    N: int = 4
    wavelengths_um: tuple[float, ...] = (1.545, 1.550, 1.555)
    alpha_scale: float = 0.45
    seed: int = 7

    # Hybrid TFLN-on-SiN stack (um)
    fiber_pitch: float = 127.0
    t_sin: float = 0.300
    t_gap: float = 0.050
    t_tfln: float = 0.300
    n_sio2: float = 1.444
    n_sin: float = 1.980
    n_tfln_o: float = 2.211
    n_tfln_e: float = 2.138
    wg_width: float = 0.800

    # Freeform design region (um)
    design_lx: float = 18.0
    design_ly: float = 20.0
    port_pitch: float = 4.0
    stub_len: float = 1.5
    # Furthest an access stub may run into the design region. The surrogate is a random
    # fill and needs several microns; an ingested Tidy3D core must present material at its
    # ports, so a multi-micron lead there is a design failure, not a rescue.
    stub_landing_max_um: float = 6.0
    stub_landing_em_max_um: float = 0.6
    dl_opt: float = 0.040
    feature_sigma_um: float = 0.500
    core_loss_db: float = 0.35

    # Foundry DRC (um)
    drc_min_width: float = 0.800
    drc_min_gap: float = 0.250
    drc_min_feature: float = 0.140
    drc_min_radius: float = 40.0
    drc_tolerance_um2: float = 0.01
    drc_defect_threshold_um2: float = 0.2
    drc_freeform_um: float = 0.14  # minimum feature of the inverse-designed core

    # GDS layer map
    layers: dict[str, tuple[int, int]] = field(default_factory=lambda: dict(DEFAULT_LAYERS))

    # Electro-optic and link parameters
    v_pi_l: float = 2.5
    mod_length_mm: float = 5.0
    mod_insertion_db: float = 1.5
    waveguide_loss_db_cm: float = 0.3
    responsivity: float = 0.85
    taper_loss_db: float = 0.08
    mux_loss_db: float = 0.60
    hybrid_excess_db: float = 0.20
    lo_boost_db: float = 6.0
    ring_radius_um: float = 12.0
    ring_q: float = 20000.0

    # System model
    baud_ghz: float = 25.0
    laser_mw_per_line: float = 4.0
    laser_wpe: float = 0.20
    tia_noise_pa_sqrt_hz: float = 12.0
    rin_db_hz: float = -145.0
    temperature_k: float = 300.0

    # Output
    artifacts_dir: Path = Path("artifacts")

    # -- derived quantities ---------------------------------------------------

    @property
    def K(self) -> int:
        return len(self.wavelengths_um)

    @property
    def freqs_hz(self) -> np.ndarray:
        return C_0 / np.asarray(self.wavelengths_um, dtype=float) / 1e-6

    @property
    def freqs_thz(self) -> np.ndarray:
        return C_0 / np.asarray(self.wavelengths_um, dtype=float) * 1e-6 / 1e12

    @property
    def v_pi(self) -> float:
        """Half-wave voltage of the full-length push-pull MZM (V)."""
        return self.v_pi_l / self.mod_length_mm * 10.0

    @property
    def phi_ref(self) -> np.ndarray:
        """Reference phase accumulated across the core at each comb line (rad)."""
        n_eff = 0.5 * (self.n_sin + self.n_tfln_o)
        return 2.0 * np.pi * n_eff / np.asarray(self.wavelengths_um) * self.design_lx

    @property
    def n_eff(self) -> float:
        return 0.5 * (self.n_sin + self.n_tfln_o)

    @property
    def lane_pitch(self) -> float:
        """Vertical pitch between adjacent input lanes (MZM stack + routing channel)."""
        return self.K * self.fiber_pitch + 250.0

    @property
    def mod_length_um(self) -> float:
        return self.mod_length_mm * 1000.0

    @property
    def nx(self) -> int:
        return int(round(self.design_lx / self.dl_opt))

    @property
    def ny(self) -> int:
        return int(round(self.design_ly / self.dl_opt))

    def layer(self, name: str) -> tuple[int, int]:
        try:
            return tuple(self.layers[name])  # type: ignore[return-value]
        except KeyError as exc:  # pragma: no cover - configuration error
            raise KeyError(f"layer {name!r} is not in the layer map: {sorted(self.layers)}") from exc

    def port_y(self, index: int, count: int | None = None) -> float:
        """Vertical coordinate of port ``index`` in a centred row of ``count`` ports."""
        count = self.N if count is None else count
        return (index - (count - 1) / 2.0) * self.port_pitch

    def path(self, artifact: str) -> Path:
        """Resolve a well-known artifact name and make sure its directory exists."""
        directory = Path(self.artifacts_dir)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / ARTIFACTS[artifact]

    def validate(self) -> None:
        """Raise ``ValueError`` if the configuration is not physically buildable."""
        if not isinstance(self.N, int) or isinstance(self.N, bool):
            raise ValueError("N must be an integer")
        if self.N < 1:
            raise ValueError("N must be >= 1")
        if self.K < 1:
            raise ValueError("at least one comb wavelength is required")
        if self.drc_min_width > self.wg_width + 1e-9:
            raise ValueError("drc_min_width must not exceed the waveguide width")

    def with_overrides(self, **changes: Any) -> "AcceleratorConfig":
        """Return a copy with validated overrides applied."""
        clean: dict[str, Any] = {}
        for key, value in changes.items():
            if value is None:
                continue
            if key == "wavelengths_um":
                value = tuple(float(w) for w in value)
            if key == "layers":
                value = {k: tuple(v) for k, v in value.items()}
            if key == "artifacts_dir":
                value = Path(value)
            clean[key] = value
        return replace(self, **clean)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Path):
                value = str(value)
            elif isinstance(value, tuple):
                value = list(value)
            elif isinstance(value, dict):
                value = {k: list(v) for k, v in value.items()}
            out[item.name] = value
        return out

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path


def load_config(path: str | Path | None = None, **overrides: Any) -> AcceleratorConfig:
    """Load a YAML config (optional) and apply keyword overrides."""
    values: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a YAML mapping")
        values.update(loaded)

    known = {item.name for item in fields(AcceleratorConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")

    config = AcceleratorConfig(**values)
    if overrides:
        config = config.with_overrides(**overrides)
    config.validate()
    return config
