from __future__ import annotations

import pytest

from rcwa_app.adapters.solver_mock.engine import MockSolverEngine
from rcwa_app.domain.models import SweepRequest
from rcwa_app.orchestration.session import default_config
from rcwa_app.plotting_plotly.presenter import PlotPresenterPlotly


@pytest.fixture(scope="session")
def small_grids():
    # Tiny grids keep figures compact and deterministic
    lam = [3.0, 4.0, 5.0]
    th = [0.0, 15.0, 30.0]
    return lam, th


@pytest.fixture(scope="session")
def small_result(small_grids):
    lam, th = small_grids
    cfg = default_config()
    req = SweepRequest(config=cfg, sweep_lambda=True,
                       lambda_grid_um=lam, theta_grid_deg=th)
    engine = MockSolverEngine()
    return engine.run(req)


@pytest.fixture(scope="session")
def presenter():
    return PlotPresenterPlotly()


@pytest.fixture(scope="session")
def spectral_fig(small_result, presenter):
    # Use the mid angle in the small grid (15 deg)
    return presenter.spectral_plot(small_result, fixed_theta=15.0)


@pytest.fixture(scope="session")
def map_fig(small_result, presenter):
    return presenter.map_eps(small_result)