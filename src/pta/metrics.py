"""System model: throughput, wall-plug power, SNR and ENOB.

The optical numbers come from the same configuration that built the chip, so the energy
model moves whenever the layout does. The model follows the standard balanced-homodyne
treatment: shot noise from the LO, TIA input-referred current noise, and residual laser RIN
suppressed by the balanced detector's common-mode rejection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .config import AcceleratorConfig

Q_E = 1.60217663e-19
K_B = 1.380649e-23


@dataclass(frozen=True)
class LinkParameters:
    v_pi: float
    laser_mw_per_line: float
    baud_ghz: float
    laser_wpe: float
    tia_noise_pa_sqrt_hz: float
    responsivity: float
    cmrr_db: float = 25.0
    electrical_pj_per_symbol: float = 2.0  # 0.8 pJ DAC + 1.2 pJ ADC
    heater_mw: float = 2.5
    rf_termination_ohm: float = 50.0


def link_parameters(cfg: AcceleratorConfig) -> LinkParameters:
    return LinkParameters(
        v_pi=cfg.v_pi,
        laser_mw_per_line=cfg.laser_mw_per_line,
        baud_ghz=cfg.baud_ghz,
        laser_wpe=cfg.laser_wpe,
        tia_noise_pa_sqrt_hz=cfg.tia_noise_pa_sqrt_hz,
        responsivity=cfg.responsivity,
    )


def evaluate(cfg: AcceleratorConfig, link: LinkParameters | None = None) -> dict[str, Any]:
    """Compute TOPS, wall-plug power, TOPS/W, SNR and ENOB for this configuration."""
    link = link or link_parameters(cfg)
    n, k = cfg.N, cfg.K
    symbol_rate = link.baud_ghz * 1e9

    macs_per_symbol = n * n * k
    tops = 2.0 * macs_per_symbol * symbol_rate / 1e12

    # -- power ------------------------------------------------------------
    p_signal_optical = n * k * link.laser_mw_per_line * 1e-3
    p_lo_optical = p_signal_optical * 10.0 ** (cfg.lo_boost_db / 10.0)
    p_laser_wall = (p_signal_optical + p_lo_optical) / link.laser_wpe

    v_rms = 0.35 * link.v_pi
    p_rf = n * k * v_rms**2 / link.rf_termination_ohm
    p_conversion = n * k * link.electrical_pj_per_symbol * 1e-12 * symbol_rate
    heaters = 2 * n * k + n
    p_thermal = heaters * link.heater_mw * 1e-3

    p_total = p_laser_wall + p_rf + p_conversion + p_thermal
    tops_per_w = tops / p_total
    energy_per_mac_fj = p_total / (macs_per_symbol * symbol_rate) * 1e15

    # -- receiver ---------------------------------------------------------
    loss_db = cfg.mux_loss_db + cfg.mod_insertion_db + cfg.taper_loss_db + cfg.hybrid_excess_db
    link_transmission = 10.0 ** (-loss_db / 10.0)

    p_signal_pd = link.laser_mw_per_line * 1e-3 * link_transmission * cfg.alpha_scale**2
    p_lo_pd = (p_lo_optical / (n * k)) * 10.0 ** (-0.6 / 10.0)

    i_peak = 2.0 * link.responsivity * np.sqrt(p_signal_pd * p_lo_pd)
    i_rms = i_peak / np.sqrt(3.0)

    bandwidth = 0.75 * symbol_rate
    i_dc_per_pd = link.responsivity * p_lo_pd / 2.0
    var_shot = 2.0 * Q_E * (2.0 * i_dc_per_pd) * bandwidth
    var_tia = (link.tia_noise_pa_sqrt_hz * 1e-12) ** 2 * bandwidth
    cmrr = 10.0 ** (-link.cmrr_db / 10.0)
    rin = 10.0 ** (cfg.rin_db_hz / 10.0)
    var_rin = cmrr * rin * (link.responsivity * p_lo_pd) ** 2 * bandwidth

    noise = float(np.sqrt(var_shot + var_tia + var_rin))
    snr_linear = (i_rms / noise) ** 2
    snr_db = 10.0 * np.log10(snr_linear)
    enob = (snr_db - 1.76) / 6.02

    return {
        "N": n,
        "K": k,
        "wavelengths_nm": [float(w * 1e3) for w in cfg.wavelengths_um],
        "baud_ghz": link.baud_ghz,
        "macs_per_symbol": macs_per_symbol,
        "tops": float(tops),
        "power_w": {
            "laser": float(p_laser_wall),
            "rf_drive": float(p_rf),
            "dac_adc": float(p_conversion),
            "thermal_tuning": float(p_thermal),
            "total": float(p_total),
        },
        "tops_per_w": float(tops_per_w),
        "energy_per_mac_fj": float(energy_per_mac_fj),
        "receiver": {
            "signal_power_per_pd_w": float(p_signal_pd),
            "lo_power_per_pd_w": float(p_lo_pd),
            "signal_current_rms_a": float(i_rms),
            "noise_current_rms_a": noise,
            "shot_noise_rms_a": float(np.sqrt(var_shot)),
            "tia_noise_rms_a": float(np.sqrt(var_tia)),
            "rin_noise_rms_a": float(np.sqrt(var_rin)),
            "link_transmission": float(link_transmission),
        },
        "snr_db": float(snr_db),
        "enob_bits": float(enob),
        "heater_count": heaters,
    }


def pareto_sweep(
    base: AcceleratorConfig,
    sizes: tuple[int, ...] = (4, 8, 16, 32),
    channels: tuple[int, ...] = (4, 8, 16),
    laser_mw: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
) -> list[dict[str, Any]]:
    """Sweep core size, comb count and laser power; returns every design point."""
    points: list[dict[str, Any]] = []
    for n in sizes:
        for k in channels:
            wavelengths = tuple(float(w) for w in np.linspace(1.535, 1.565, k))
            for power in laser_mw:
                scaled = replace(
                    base,
                    N=n,
                    wavelengths_um=wavelengths,
                    laser_mw_per_line=power,
                    ring_q=base.ring_q,
                )
                metrics = evaluate(scaled)
                points.append(
                    {
                        "N": n,
                        "K": k,
                        "laser_mw_per_line": power,
                        "tops": metrics["tops"],
                        "power_w": metrics["power_w"]["total"],
                        "tops_per_w": metrics["tops_per_w"],
                        "enob_bits": metrics["enob_bits"],
                    }
                )
    return points


def pareto_front(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-dominated points maximising both TOPS/W and ENOB."""
    front: list[dict[str, Any]] = []
    for point in sorted(points, key=lambda item: item["enob_bits"], reverse=True):
        if not front or point["tops_per_w"] > max(item["tops_per_w"] for item in front):
            front.append(point)
    return list(reversed(front))
