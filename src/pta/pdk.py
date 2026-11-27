"""PhotonForge technology bindings for the hybrid TFLN-on-SiN stack.

The foundry layer map, the 800 nm single-mode SiN port specification and the 3D extrusion
used for Tidy3D hand-off all live here, so the GDS layers and the electromagnetic model can
never drift apart.
"""

from __future__ import annotations

from functools import lru_cache

import photonforge as pf
import tidy3d as td

from .config import AcceleratorConfig

PORT_SPEC = "SiN"


@lru_cache(maxsize=8)
def _technology_cached(
    layer_items: tuple[tuple[str, tuple[int, int]], ...],
    wg_width: float,
    t_sin: float,
    n_sin: float,
    n_sio2: float,
    n_tfln_o: float,
    t_gap: float,
    t_tfln: float,
    min_width: float,
    min_gap: float,
) -> pf.Technology:
    layers = {name: layer for name, layer in layer_items}
    sin = layers["sin_core"]
    tfln = layers["tfln_slab"]

    layer_specs = {
        "SiN": pf.LayerSpec(sin, "Etched SiN core", "#3b82f6", conductive=False, min_width=min_width, min_spacing=min_gap),
        "TFLN": pf.LayerSpec(tfln, "Bonded X-cut TFLN slab", "#f97316", conductive=False, min_width=min_width, min_spacing=min_gap),
        "HEATER": pf.LayerSpec(layers["heater"], "TiN thermo-optic heater", "#ef4444", conductive=True, min_width=1.0, min_spacing=1.0),
        "RF": pf.LayerSpec(layers["rf_metal"], "Au GSG travelling-wave electrode", "#eab308", conductive=True, min_width=4.0, min_spacing=4.0),
        "PAD": pf.LayerSpec(layers["oxide_open"], "Passivation opening over pads", "#22c55e", conductive=False),
        "MARKER": pf.LayerSpec(layers["exclude"], "Device marker / DRC exclusion", "#94a3b8", conductive=False),
    }

    port_spec = pf.PortSpec(
        description="SiN 800 nm single-mode strip",
        width=wg_width,
        limits=(-t_sin / 2, t_sin / 2),
        num_modes=1,
        polarization="TE",
        target_neff=0.5 * (n_sin + 1.85),
        path_profiles=[(wg_width, 0.0, sin)],
    )

    extrusion_specs = [
        pf.ExtrusionSpec(
            pf.MaskSpec(sin),
            td.Medium(permittivity=n_sin**2),
            (-t_sin / 2, t_sin / 2),
        ),
        pf.ExtrusionSpec(
            pf.MaskSpec(tfln),
            td.Medium(permittivity=n_tfln_o**2),
            (t_sin / 2 + t_gap, t_sin / 2 + t_gap + t_tfln),
        ),
    ]

    return pf.Technology(
        "PTA TFLN-on-SiN",
        "1.0",
        layers=layer_specs,
        extrusion_specs=extrusion_specs,
        ports={PORT_SPEC: port_spec},
        background_medium=td.Medium(permittivity=n_sio2**2),
    )


def technology(cfg: AcceleratorConfig) -> pf.Technology:
    """Build (and cache) the PhotonForge technology for this configuration."""
    return _technology_cached(
        tuple(
            sorted(
                (name, cfg.layer(name))
                for name in ("sin_core", "tfln_slab", "heater", "rf_metal", "oxide_open", "exclude")
            )
        ),
        cfg.wg_width,
        cfg.t_sin,
        cfg.n_sin,
        cfg.n_sio2,
        cfg.n_tfln_o,
        cfg.t_gap,
        cfg.t_tfln,
        cfg.drc_min_width,
        cfg.drc_min_gap,
    )


def install(cfg: AcceleratorConfig) -> pf.Technology:
    """Make this technology the process-wide default and return it.

    ``pf.Port(..., spec="SiN")`` resolves port specifications against
    ``photonforge.config.default_technology``, so the default has to be installed before any
    component is built.
    """
    tech = technology(cfg)
    pf.config.default_technology = tech
    return tech


def port(cfg: AcceleratorConfig, center, direction: float, name: str | None = None) -> pf.Port:
    """Create a SiN strip port; ``direction`` is the angle of power entering the device."""
    return pf.Port(center=center, input_direction=direction, spec=PORT_SPEC)
