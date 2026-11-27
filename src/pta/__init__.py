"""Photonic tensor accelerator (PTA) golden build.

A coherent frequency-comb, signed-weight photonic matrix-vector multiplier built from a
Tidy3D inverse-designed freeform SiN core bonded to an X-cut TFLN slab, assembled into a
full tapeout layout in PhotonForge.

Pipeline stages (see ``docs/implementation-plan.md``):

``design``   target weights -> freeform core S-matrix (+ optional Tidy3D job spec)
``import-em`` Tidy3D density + monitor amplitudes -> core_smatrix.npz
``layout``   core polygons -> DRC-clean PhotonForge chip -> GDSII
``verify``   circuit-level coherent homodyne matrix-vector multiplication check
``metrics``  TOPS/W, SNR and ENOB system model
``bundle``   portal upload manifest + zip archive
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import AcceleratorConfig, load_config

__all__ = ["AcceleratorConfig", "load_config", "__version__"]
