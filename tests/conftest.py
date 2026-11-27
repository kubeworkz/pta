from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pta.config import load_config  # noqa: E402
from pta.pipeline import run_pipeline  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "pta.yaml"


@pytest.fixture(scope="session")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def small_cfg(tmp_path_factory):
    return load_config(
        CONFIG_PATH,
        N=2,
        wavelengths_um=[1.55, 1.56],
        artifacts_dir=tmp_path_factory.mktemp("pta_artifacts"),
    )


@pytest.fixture(scope="session")
def pipeline_result(small_cfg):
    return run_pipeline(small_cfg, run_drc=False)
