from __future__ import annotations

import numpy as np
import pytest

from rcwa_app.adapters.solver_mock.engine import MockSolverEngine
from rcwa_app.domain.models import SweepRequest
from rcwa_app.orchestration.session import default_config

xr = pytest.importorskip("xarray")


def test_mock_engine_respects_dataset_contracts() -> None:
    # Tiny, explicit grids (monotonic, small)
    lam = [8.0, 10.0, 12.0]  # μm
    th = [0.0, 30.0]  # degrees

    cfg = default_config()
    req = SweepRequest(
        config=cfg,
        sweep_lambda=True,
        lambda_grid_um=lam,
        theta_grid_deg=th,
    )

    result = MockSolverEngine().run(req)
    ds = result.data  # xarray.Dataset

    # Coordinates & dims (per README/contracts)
    for coord in ("lambda_um", "theta_deg"):
        assert coord in ds.coords, f"Missing coord {coord}"
        assert coord in ds.dims, f"Missing dim {coord}"

    # Minimal required variable for the current mock: eps
    assert "eps" in ds.data_vars, "Mock must provide eps(λ,θ)"

    # Physical bounds on eps
    eps = ds["eps"].values
    assert np.isfinite(eps).all()
    assert ((eps >= 0) & (eps <= 1)).all()

    # Optional energy accounting check if R/T/A are present
    opt = [v for v in ("Rsum", "Tsum", "Asum") if v in ds.data_vars]
    if len(opt) == 3:
        energy = ds["Rsum"] + ds["Tsum"] + ds["Asum"]
        assert float(np.nanmax(np.abs(energy.values - 1.0))) <= 5e-3

    # Monotonic grids
    assert np.all(np.diff(ds["lambda_um"].values) > 0)
    assert np.all(np.diff(ds["theta_deg"].values) >= 0)
