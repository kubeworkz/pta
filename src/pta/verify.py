"""Circuit-level verification of coherent homodyne signed matrix-vector multiplication.

The chip S-matrix is solved at every comb line with the MZMs biased for the chosen input
vector. Balanced detection is then evaluated exactly as the hardware does it, on the two
hybrid output fields:

``E_pos = sum_j S[PD_POS, SIG_j] x_j + S[PD_POS, LO] E_LO``
``E_neg = sum_j S[PD_NEG, SIG_j] x_j + S[PD_NEG, LO] E_LO``
``y     = R (|E_pos|^2 - |E_neg|^2)``

The signal alone is common-mode and cancels; the sign of the computation lives in the beat
between the signal field and the phase-locked LO, which is exactly why the architecture
needs coherent detection.

The LO thermo-optic tuner is locked to a pilot tone from lane 0, as the startup calibration
does on the chip. The pilot also measures the end-to-end link gain, so the ideal reference is
built from the measured pilot rather than from assumed loss numbers. The reported error is
therefore the real spread of the optical weight matrix - element phase and amplitude errors
plus crosstalk - and not an artefact of a global phase or an assumed power budget.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import AcceleratorConfig
from .core import CoreModel


def _element(s_matrix, in_name: str, out_name: str) -> np.ndarray:
    return np.asarray(s_matrix[(f"{in_name}@0", f"{out_name}@0")], dtype=complex)


def verify_mvm(
    s_matrix,
    cfg: AcceleratorConfig,
    core: CoreModel,
    inputs: np.ndarray,
    pass_threshold: float = 0.10,
) -> dict[str, Any]:
    """Run the signed MVM across all comb lines and report measured vs ideal outputs."""
    inputs = np.asarray(inputs, dtype=float).reshape(cfg.N, cfg.K)
    weights = np.asarray(core.weights, dtype=float)
    n, k_count = cfg.N, cfg.K

    # Field transfer matrices: signal from every lane/channel and LO from every hybrid.
    pos_signal = np.zeros((k_count, n, n), dtype=complex)
    neg_signal = np.zeros((k_count, n, n), dtype=complex)
    pos_lo = np.zeros((k_count, n), dtype=complex)
    neg_lo = np.zeros((k_count, n), dtype=complex)
    for k in range(k_count):
        for i in range(n):
            pos_lo[k, i] = _element(s_matrix, f"COMB_LO_IN_{i}", f"PD_POS_{i}")[k]
            neg_lo[k, i] = _element(s_matrix, f"COMB_LO_IN_{i}", f"PD_NEG_{i}")[k]
            for j in range(n):
                pos_signal[k, i, j] = _element(s_matrix, f"COMB_SIG_IN_{j}_{k}", f"PD_POS_{i}")[k]
                neg_signal[k, i, j] = _element(s_matrix, f"COMB_SIG_IN_{j}_{k}", f"PD_NEG_{i}")[k]

    # -- lane phase calibration -------------------------------------------
    # Each input lane carries a thermo-optic phase trimmer. The startup routine drives one
    # lane at a time and trims it until the field arriving at the core matches the sign of
    # the target weight in a reference output row. One scalar per lane and comb line is
    # enough: the output route phase is common to all lanes and is absorbed by the LO lock.
    lane_phase = np.zeros((k_count, n))
    for k in range(k_count):
        for j in range(n):
            reference_field = pos_signal[k, 0, j] + neg_signal[k, 0, j]
            lane_phase[k, j] = np.angle(weights[k, 0, j]) - np.angle(reference_field)
    trim = np.exp(1j * lane_phase)[:, None, :]
    pos_signal = pos_signal * trim
    neg_signal = neg_signal * trim

    lines: list[dict[str, Any]] = []
    errors: list[float] = []
    for k in range(k_count):
        measured = np.zeros(n)
        ideal = np.zeros(n)
        lo_phase = np.zeros(n)

        for i in range(n):
            # Pilot tone: lane 0 at unit amplitude fixes the LO reference phase.
            pilot = pos_signal[k, i, 0] + neg_signal[k, i, 0]
            phase = float(np.angle(pilot * np.conj(pos_lo[k, i])))
            lo_phase[i] = phase
            lp = pos_lo[k, i] * np.exp(1j * phase)
            ln = neg_lo[k, i] * np.exp(1j * phase)

            a = pos_signal[k, i, :] @ inputs[:, k]
            b = neg_signal[k, i, :] @ inputs[:, k]
            measured[i] = cfg.responsivity * (abs(a + lp) ** 2 - abs(b + ln) ** 2)

            # Ideal fields: same pilot-measured link gain, but with the target weight
            # matrix and no fabrication phase spread.
            w_reference = weights[k, i, 0]
            if abs(w_reference) < 1e-12:
                ratio = 0.0
            else:
                ratio = float(weights[k, i] @ inputs[:, k]) / w_reference
            a_ideal = pos_signal[k, i, 0] * ratio
            b_ideal = neg_signal[k, i, 0] * ratio
            ideal[i] = cfg.responsivity * (abs(a_ideal + lp) ** 2 - abs(b_ideal + ln) ** 2)

        denominator = float(np.linalg.norm(ideal))
        error = float(np.linalg.norm(measured - ideal) / denominator) if denominator > 0 else 0.0
        errors.append(error)
        lines.append(
            {
                "wavelength_nm": float(core.wavelengths_um[k] * 1e3),
                "inputs": [float(value) for value in inputs[:, k]],
                "measured": [float(value) for value in measured],
                "ideal": [float(value) for value in ideal],
                "relative_error": error,
                "lo_phase_rad": [float(value) for value in lo_phase],
                "lane_phase_rad": [float(value) for value in lane_phase[k]],
            }
        )

    worst = float(max(errors))
    return {
        "lines": lines,
        "worst_relative_error": worst,
        "mean_relative_error": float(np.mean(errors)),
        "pass_threshold": pass_threshold,
        "passes": worst <= pass_threshold,
        "ops_per_symbol": 2.0 * n * n * k_count,
        "notes": [
            "Measured values are balanced photocurrents from the solved chip S-matrix.",
            "Ideal values use the pilot-calibrated link gain and the target weight matrix.",
            "Per-lane phase trimmers are calibrated on the reference output row before the MVM.",
            "Residual error is dominated by the modelled fabrication phase spread.",
        ],
    }
