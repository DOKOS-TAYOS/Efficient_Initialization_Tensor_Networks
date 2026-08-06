import math
from pathlib import Path

import pytest
import torch
from streamlit.testing.v1 import AppTest

from normalizer_module import MPO, MPS


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _assert_finite_unit_norm(layer: MPS | MPO) -> None:
    norm = layer.compute_frob()

    assert math.isfinite(norm)
    assert norm == pytest.approx(1.0, rel=1e-5)
    assert all(torch.isfinite(node.tensor).all().item() for node in layer.layer_nodes)


def test_mps_initialization_produces_finite_unit_norm() -> None:
    torch.manual_seed(0)

    _assert_finite_unit_norm(MPS(n_nodes=3, phys_dim=2, bond_dim=2))


def test_mpo_initialization_produces_finite_unit_norm() -> None:
    torch.manual_seed(0)

    _assert_finite_unit_norm(MPO(n_nodes=3, phys_dim=2, bond_dim=2))


def test_streamlit_demo_loads_without_exceptions() -> None:
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
