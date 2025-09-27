from __future__ import annotations

import numpy as np
import pandas as pd

from rcwa_app.adapters.solver_mock.engine import MockSolverEngine
from rcwa_app.domain.models import SweepRequest
from rcwa_app.orchestration.session import default_config
from rcwa_app.plotting_plotly.presenter import PlotPresenterPlotly


def test_overlay_has_two_traces() -> None:
    cfg = default_config()
    req = SweepRequest(
        config=cfg,
        sweep_lambda=True,
        lambda_grid_um=[3.0, 4.0, 5.0],
        theta_grid_deg=[0.0, 15.0, 30.0],
    )
    res = MockSolverEngine().run(req)

    lam = res.data["lambda_um"].values
    i = int(np.argmin(np.abs(res.data["theta_deg"].values - 15.0)))
    y = res.data["eps"].isel(theta_deg=i).values
    df_ref = pd.DataFrame({"lambda_um": lam, "eps": y})

    fig = PlotPresenterPlotly().spectral_overlay(res, 15.0, df_ref, ref_name="unit-test")
    # one model trace + one reference trace
    assert len(fig.data) == 2
    names = [tr.name for tr in fig.data]
    assert any(n.lower().startswith("ε (unpolarized)".lower()) for n in names)
    assert any("ref: unit-test" in n for n in names)
