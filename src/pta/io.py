"""S-matrix interchange files.

A ``.npz`` with a documented layout is the hand-off between the PhotonForge solve, the
verification report and the portal bundle:

``freqs_hz``  ``(K,)``
``ports``     ``(P,)`` port names in matrix order
``s_matrix``  ``(K, P, P)`` complex, indexed ``[line, out, in]``
``meta``      JSON string with build metadata
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def port_name(key: str) -> str:
    return key.split("@", 1)[0]


def save_smatrix(
    path: str | Path,
    s_matrix,
    freqs_hz: np.ndarray,
    meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(s_matrix.ports)
    index = {name: i for i, name in enumerate(names)}
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    array = np.zeros((len(freqs_hz), len(names), len(names)), dtype=complex)
    for (in_key, out_key), values in s_matrix.elements.items():
        source = port_name(in_key)
        target = port_name(out_key)
        if source not in index or target not in index:
            continue
        array[:, index[target], index[source]] = np.asarray(values, dtype=complex)
    np.savez_compressed(
        path,
        freqs_hz=freqs_hz,
        ports=np.asarray(names),
        s_matrix=array,
        meta=np.asarray(json.dumps(meta or {}, default=str)),
    )
    return path


def load_smatrix(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    data = np.load(Path(path), allow_pickle=False)
    return (
        np.asarray(data["freqs_hz"], dtype=float),
        [str(name) for name in data["ports"]],
        np.asarray(data["s_matrix"], dtype=complex),
        json.loads(str(data["meta"])) if "meta" in data else {},
    )
