"""Scattering-matrix models for every building block of the accelerator.

Each block is a real piece of GDS geometry carrying a :class:`photonforge.DataModel` whose
S-matrix encodes its optical behaviour at the comb lines. Keeping the models here (rather
than inside the geometry builders) means the same numbers feed the circuit solver, the
verification report and the system model.

Block status is recorded so tapeout readiness is never overstated:

===============  ==========================================================
``freeform``     EM-verified by Tidy3D (S-matrix ingested from the solve)
``mux``          analytic add-drop Lorentzian; geometry needs FDTD sign-off
``mzm``          analytic push-pull response; geometry needs EO sign-off
``hybrid``       ideal 2x2; geometry needs FDTD sign-off
``route``        propagation phase + loss at the configured loss figure
===============  ==========================================================
"""

from __future__ import annotations

import numpy as np
import photonforge as pf

from .config import AcceleratorConfig


def smatrix(
    port_names: list[str],
    elements: dict[tuple[str, str], np.ndarray],
    freqs_hz: np.ndarray,
) -> pf.SMatrix:
    """Build a PhotonForge S-matrix from ``(in@mode, out@mode)`` element arrays."""
    return pf.SMatrix(
        frequencies=np.asarray(freqs_hz, dtype=float),
        elements={key: np.asarray(value, dtype=complex) for key, value in elements.items()},
        ports={name: None for name in port_names},
    )


def data_model(
    port_names: list[str],
    elements: dict[tuple[str, str], np.ndarray],
    freqs_hz: np.ndarray,
) -> pf.DataModel:
    return pf.DataModel(s_matrix=smatrix(port_names, elements, freqs_hz))


def two_port(a: str, b: str, transfer: np.ndarray, freqs_hz: np.ndarray) -> pf.DataModel:
    """Reciprocal two-port model with the given field transfer ``a -> b``."""
    transfer = np.asarray(transfer, dtype=complex)
    return data_model([a, b], {(f"{a}@0", f"{b}@0"): transfer, (f"{b}@0", f"{a}@0"): transfer}, freqs_hz)


# -- freeform core -----------------------------------------------------------


def core_model(core, freqs_hz: np.ndarray) -> pf.DataModel:
    """Reciprocal ``2N``-port model of the inverse-designed core."""
    n = core.N
    names = [f"IN_{i}" for i in range(n)] + [f"OUT_{i}" for i in range(n)]
    elements: dict[tuple[str, str], np.ndarray] = {}
    for out_idx, out_name in enumerate(names):
        for in_idx, in_name in enumerate(names):
            elements[(f"{in_name}@0", f"{out_name}@0")] = core.s_matrix[:, out_idx, in_idx]
    return data_model(names, elements, np.asarray(freqs_hz, dtype=float))


# -- TFLN push-pull MZM ------------------------------------------------------


def mzm_transfer(cfg: AcceleratorConfig, voltages: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
    """Field transmission of one null-biased push-pull MZM per comb line.

    ``E_out / E_in = sin(pi V / (2 V_pi))`` gives a bipolar field in ``[-1, 1]``, so the
    sign of the input is encoded in a 0/pi optical phase and no DC bias is needed.
    """
    voltages = np.atleast_1d(np.asarray(voltages, dtype=float))
    amplitude = np.sin(np.pi * voltages / (2.0 * cfg.v_pi))
    loss = 10.0 ** (-(cfg.mod_insertion_db + cfg.taper_loss_db) / 20.0)
    wavelength = 299792458.0 / np.asarray(freqs_hz, dtype=float)
    phase = 2.0 * np.pi * cfg.n_eff * cfg.mod_length_um / (np.asarray(wavelength) * 1e-6)
    return amplitude * loss * np.exp(1j * phase)


# -- add-drop microring WDM multiplexer -------------------------------------


def channel_response(
    cfg: AcceleratorConfig,
    center_wavelength_um: float,
    freqs_hz: np.ndarray,
) -> np.ndarray:
    """Add-drop Lorentzian of one ring on the dense frequency grid."""
    wavelength = 299792458.0 / np.asarray(freqs_hz, dtype=float) * 1e6
    fwhm = center_wavelength_um / cfg.ring_q
    detuning = wavelength - center_wavelength_um
    return 1.0 / (1.0 + 2.0j * detuning / fwhm)


def mux_model(cfg: AcceleratorConfig, freqs_hz: np.ndarray) -> pf.DataModel:
    """``K`` add ports to one through bus.

    Channel ``k`` is dropped from ``CH_IN_k`` onto ``BUS_OUT`` with the ring's Lorentzian
    response, an insertion phase and the configured excess loss.
    """
    loss = 10.0 ** (-cfg.mux_loss_db / 20.0)
    names = [f"CH_IN_{k}" for k in range(cfg.K)] + ["BUS_OUT"]
    elements: dict[tuple[str, str], np.ndarray] = {}
    for k, wavelength in enumerate(cfg.wavelengths_um):
        response = loss * channel_response(cfg, wavelength, freqs_hz)
        # Propagation delay of the drop path at the channel wavelength.
        response = response * np.exp(2.0j * np.pi * cfg.n_eff * 40.0 / wavelength)
        elements[(f"CH_IN_{k}@0", "BUS_OUT@0")] = response
        elements[("BUS_OUT@0", f"CH_IN_{k}@0")] = response
    return data_model(names, elements, freqs_hz)


# -- balanced homodyne hybrid ------------------------------------------------


def hybrid_model(cfg: AcceleratorConfig, freqs_hz: np.ndarray) -> pf.DataModel:
    """Ideal 180 degree 2x2 hybrid with excess loss.

    ``PD_POS = (SIG + LO)/sqrt(2)`` and ``PD_NEG = (SIG - LO)/sqrt(2)`` so that subtracting
    the two photocurrents yields ``2 R Re(E_sig E_LO*)``.
    """
    amp = np.full(len(freqs_hz), 10.0 ** (-cfg.hybrid_excess_db / 20.0) / np.sqrt(2.0), dtype=complex)
    names = ["SIG_IN", "LO_IN", "PD_POS", "PD_NEG"]
    pairs = [
        ("SIG_IN", "PD_POS", +1.0),
        ("LO_IN", "PD_POS", +1.0),
        ("SIG_IN", "PD_NEG", +1.0),
        ("LO_IN", "PD_NEG", -1.0),
    ]
    elements: dict[tuple[str, str], np.ndarray] = {}
    for src, dst, sign in pairs:
        elements[(f"{src}@0", f"{dst}@0")] = sign * amp
        elements[(f"{dst}@0", f"{src}@0")] = sign * amp
    return data_model(names, elements, freqs_hz)


# -- LO distribution ---------------------------------------------------------


def lo_splitter_model(cfg: AcceleratorConfig, count: int, freqs_hz: np.ndarray) -> pf.DataModel:
    """Ideal 1 x N MMI star coupler: equal amplitude, equal phase on every leaf."""
    amp = np.full(len(freqs_hz), 10.0 ** (-0.2 / 20.0) / np.sqrt(count), dtype=complex)
    names = ["LO_IN"] + [f"LO_OUT_{i}" for i in range(count)]
    elements: dict[tuple[str, str], np.ndarray] = {}
    for i in range(count):
        elements[("LO_IN@0", f"LO_OUT_{i}@0")] = amp
        elements[(f"LO_OUT_{i}@0", "LO_IN@0")] = amp
    return data_model(names, elements, freqs_hz)


# -- waveguide route ---------------------------------------------------------


def route_model(
    cfg: AcceleratorConfig,
    length_um: float,
    freqs_hz: np.ndarray,
    ports: tuple[str, str] = ("P0", "P1"),
) -> pf.DataModel:
    """Propagation-only model for a routed waveguide of the given physical length."""
    loss = 10.0 ** (-cfg.waveguide_loss_db_cm * length_um * 1e-4 / 10.0)
    wavelength = 299792458.0 / np.asarray(freqs_hz, dtype=float) * 1e6
    phase = 2.0 * np.pi * cfg.n_eff * length_um / wavelength
    transfer = loss * np.exp(1j * phase)
    return two_port(ports[0], ports[1], transfer, freqs_hz)


def dense_freq_grid(cfg: AcceleratorConfig, pad_nm: float = 6.0, step_nm: float = 0.05) -> np.ndarray:
    """Frequency grid that resolves the ring lineshapes and contains every comb line."""
    wavelengths = np.arange(
        min(cfg.wavelengths_um) - pad_nm * 1e-3,
        max(cfg.wavelengths_um) + pad_nm * 1e-3 + 0.5 * step_nm * 1e-3,
        step_nm * 1e-3,
    )
    wavelengths = np.unique(np.concatenate([wavelengths, np.asarray(cfg.wavelengths_um, dtype=float)]))
    return 299792458.0 / wavelengths * 1e6
