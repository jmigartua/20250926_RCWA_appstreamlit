from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rcwa_app.adapters.solver_mock.engine import MockSolverEngine
from rcwa_app.domain.models import SweepRequest
from rcwa_app.orchestration.session import default_config
from rcwa_app.validation.metrics import rmse_eps_on_common_lambda


def _result_small() -> Any:
    cfg = default_config()
    req = SweepRequest(
        config=cfg,
        sweep_lambda=True,
        lambda_grid_um=[3.0, 4.0, 5.0, 6.0],
        theta_grid_deg=[0.0, 15.0, 30.0],
    )
    return MockSolverEngine().run(req)


def test_rmse_matches_injected_noise() -> None:
    res = _result_small()
    # Take model line at θ=15°, then inject tiny deterministic noise
    ds = res.data
    lam = ds["lambda_um"].values
    i = int(np.argmin(np.abs(ds["theta_deg"].values - 15.0)))
    y = ds["eps"].isel(theta_deg=i).values
    noise = np.array([0.005, -0.005, 0.005, -0.005], dtype=float)
    y_ref = (y + noise).clip(0.0, 1.0)
    df_ref = pd.DataFrame({"lambda_um": lam, "eps": y_ref})

    rmse = rmse_eps_on_common_lambda(ds, theta_deg=15.0, ref=df_ref)
    # Expected RMSE is sqrt(mean(noise^2)) when no clipping occurs
    expected = float(np.sqrt(np.mean((y - y_ref) ** 2)))
    assert abs(rmse - expected) < 1e-6
