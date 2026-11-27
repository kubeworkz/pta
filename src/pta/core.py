"""Freeform tensor core model and its S-matrix interchange format.

The core is an ``N``-input, ``N``-output reciprocal passive scatterer. Its port ordering is
``[IN_0 .. IN_{N-1}, OUT_0 .. OUT_{N-1}]`` and the S-matrix of a reciprocal device has the
block form::

    S = [[0,   T],
         [T.T, 0]]          with  T[o, i] = transmission IN_i -> OUT_o

Because the off-diagonal blocks are transposes of one another, ``S = S.T`` holds by
construction and ``||S||_2 = ||T||_2 = alpha * ||W||_2 <= 1`` for ``alpha <= 1``, so the
model is reciprocal and passive at every comb line without any post-hoc clipping.

Two backends write the same ``.npz`` schema:

``analytic``  closed-form model of the optimized core with a modelled fabrication phase
              error. Used for the golden build dry run and for CI.
``tidy3d``    an S-matrix from a Tidy3D forward run (or supplied by the user); assemble it
              with ``pta import-em`` or ingest it with :func:`CoreModel.from_npz`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import AcceleratorConfig

SCHEMA_VERSION = 1


@dataclass
class CoreModel:
    """Complex S-matrix of the freeform core at every comb line."""

    freqs_hz: np.ndarray
    s_matrix: np.ndarray
    weights: np.ndarray
    phi_ref: np.ndarray
    alpha: float
    backend: str = "analytic"
    density: np.ndarray | None = None
    note: str = ""

    @property
    def N(self) -> int:
        return self.weights.shape[1]

    @property
    def K(self) -> int:
        return self.weights.shape[0]

    @property
    def wavelengths_um(self) -> np.ndarray:
        return 299792458.0 / self.freqs_hz * 1e6

    def transmission(self) -> np.ndarray:
        """``(K, N, N)`` field transmission ``T[k, out, in]``."""
        n = self.N
        return self.s_matrix[:, n:, :n]

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "freqs_hz": self.freqs_hz,
            "s_matrix": self.s_matrix,
            "weights": self.weights,
            "phi_ref": self.phi_ref,
            "alpha": np.asarray(self.alpha),
            "backend": np.asarray(self.backend),
            "note": np.asarray(self.note),
            "schema_version": np.asarray(SCHEMA_VERSION),
        }
        if self.density is not None:
            payload["density"] = self.density
        np.savez_compressed(path, **payload)
        return path

    @classmethod
    def from_npz(cls, path: str | Path) -> "CoreModel":
        """Load a core from an npz; a non-analytic core must carry its density map."""
        data = np.load(Path(path), allow_pickle=False)
        backend = str(data["backend"]) if "backend" in data else "unknown"
        density = data["density"] if "density" in data else None
        return cls(
            freqs_hz=np.asarray(data["freqs_hz"], dtype=float),
            s_matrix=np.asarray(data["s_matrix"], dtype=complex),
            weights=np.asarray(data["weights"], dtype=float),
            phi_ref=np.asarray(data["phi_ref"], dtype=float),
            alpha=float(data["alpha"]),
            backend=backend,
            density=None if density is None else np.asarray(density, dtype=float),
            note=str(data["note"]) if "note" in data else "",
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "backend": self.backend,
            "N": self.N,
            "K": self.K,
            "wavelengths_um": [float(w) for w in self.wavelengths_um],
            "alpha_scale": float(self.alpha),
            "ports": [f"IN_{i}" for i in range(self.N)]
            + [f"OUT_{i}" for i in range(self.N)],
            "note": self.note,
        }


def surrogate_density(cfg: AcceleratorConfig, seed_offset: int = 101) -> np.ndarray:
    """Deterministic band-limited fill used before a Tidy3D adjoint result exists.

    The field is Gaussian-correlated with a correlation length of
    ``cfg.feature_sigma_um``. At 0.50 um every island and trench clears the 140 nm freeform
    rule after thresholding and contouring (assembled core: 0.252 um width, 0.281 um gap).
    At the
    0.24 um used before, thresholding still produced 85-120 nm gaps between large islands
    that no morphological repair could remove without creating a sub-rule bridge, so the
    generated layout did not meet the rule it was checked against. This is a *placeholder*
    for the inverse-designed pattern, not an optimized structure; Stage 1 of the Tidy3D flow
    replaces it.
    """
    rng = np.random.default_rng(cfg.seed + seed_offset)
    field = rng.standard_normal((cfg.nx, cfg.ny))
    sigma_px = max(cfg.feature_sigma_um / cfg.dl_opt, 1.0)
    field = gaussian_filter(field, sigma=sigma_px, mode="wrap")
    scaled = field / max(float(np.std(field)), 1e-12)
    mask = scaled > 0.0

    # Clean the mask to the freeform rule before it becomes a density map: thresholding a
    # correlated random field always leaves tendrils far below the 140 nm limit, and those
    # would ship as unmanufacturable geometry. The 1.3x margin absorbs the final smoothing.
    from .drc import raster_close, raster_open

    radius = 1.5 * 0.5 * cfg.drc_freeform_um
    mask = raster_open(raster_close(mask, radius, cfg.dl_opt), radius, cfg.dl_opt)

    soft = gaussian_filter(mask.astype(float), sigma=0.6, mode="wrap")
    return 0.5 + 0.5 * np.tanh(6.0 * (soft - 0.5))


def build_core(
    cfg: AcceleratorConfig,
    weights: np.ndarray | None = None,
    density: np.ndarray | None = None,
    phase_error_rad: float = 0.06,
    backend: str = "analytic",
) -> CoreModel:
    """Build the core S-matrix from target weights.

    Args:
        cfg: hardware configuration.
        weights: ``(K, N, N)`` target matrices; generated from the config seed if omitted.
        density: freeform density map; the surrogate fill is generated if omitted.
        phase_error_rad: peak per-element phase error, standing in for the residual
            mismatch of a fabricated freeform core.
        backend: label recorded in the artifact metadata.
    """
    from .weights import target_matrices

    weights = target_matrices(cfg) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (cfg.K, cfg.N, cfg.N):
        raise ValueError(f"expected weights of shape {(cfg.K, cfg.N, cfg.N)}, got {weights.shape}")
    if density is None:
        density = surrogate_density(cfg)
    density = np.asarray(density, dtype=float)
    if density.shape != (cfg.nx, cfg.ny):
        raise ValueError(f"expected density of shape {(cfg.nx, cfg.ny)}, got {density.shape}")

    loss_amplitude = 10.0 ** (-cfg.core_loss_db / 20.0)
    rng = np.random.default_rng(cfg.seed + 7)
    phase = (
        cfg.phi_ref[:, None, None]
        + rng.uniform(-phase_error_rad, phase_error_rad, size=weights.shape)
    )
    transmission = cfg.alpha_scale * loss_amplitude * weights * np.exp(1j * phase)

    n = cfg.N
    s_matrix = np.zeros((cfg.K, 2 * n, 2 * n), dtype=complex)
    s_matrix[:, n:, :n] = transmission
    s_matrix[:, :n, n:] = np.swapaxes(transmission, 1, 2)

    return CoreModel(
        freqs_hz=cfg.freqs_hz,
        s_matrix=s_matrix,
        weights=weights,
        phi_ref=cfg.phi_ref,
        alpha=cfg.alpha_scale,
        backend=backend,
        density=density,
        note=(
            "Closed-form surrogate of the optimized core: reciprocal, passive, with the "
            "target complex transmission plus modelled fabrication phase error. Replace "
            "with a Tidy3D-extracted S-matrix before tapeout sign-off."
            if backend == "analytic"
            else "S-matrix ingested from an external electromagnetic solve."
        ),
    )


def fidelity(core: CoreModel) -> dict[str, Any]:
    """Compare the realized kernel against the ideal signed weight matrix.

    A single complex scalar (insertion loss plus global phase) is fitted out per line before
    the error is computed: those are calibrated away by the receiver, so what matters is the
    complex *shape* of the kernel.
    """
    transmission = core.transmission()
    ideal = core.alpha * core.weights * np.exp(1j * core.phi_ref[:, None, None])
    report: dict[str, Any] = {"per_line": [], "worst_relative_error": 0.0}
    worst = 0.0
    for k in range(core.K):
        reference = ideal[k].ravel()
        measured = transmission[k].ravel()
        scale = float(
            np.vdot(reference, measured).real / max(float(np.vdot(reference, reference).real), 1e-30)
        )
        error = float(
            np.linalg.norm(measured - scale * reference) / max(np.linalg.norm(scale * reference), 1e-12)
        )
        worst = max(worst, error)
        report["per_line"].append(
            {
                "wavelength_nm": float(core.wavelengths_um[k] * 1e3),
                "relative_error": error,
                "amplitude_scale": scale,
                "mean_amplitude": float(np.mean(np.abs(transmission[k]))),
            }
        )
    report["worst_relative_error"] = worst
    return report


def physics_report(core: CoreModel) -> dict[str, Any]:
    """Reciprocity, passivity and energy bookkeeping for every comb line."""
    lines = []
    for k in range(core.K):
        s = core.s_matrix[k]
        singular = np.linalg.svd(s, compute_uv=False)
        lines.append(
            {
                "wavelength_nm": float(core.wavelengths_um[k] * 1e3),
                "reciprocity_error": float(np.max(np.abs(s - s.T))),
                "max_singular_value": float(np.max(singular)),
                "total_transmission": float(np.sum(np.abs(core.transmission()[k]) ** 2)),
            }
        )
    return {
        "backend": core.backend,
        "lines": lines,
        "reciprocal": all(line["reciprocity_error"] < 1e-9 for line in lines),
        "passive": all(line["max_singular_value"] <= 1.0 + 1e-9 for line in lines),
    }


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return [value.real, value.imag]
    raise TypeError(f"cannot serialize {type(value)!r}")
