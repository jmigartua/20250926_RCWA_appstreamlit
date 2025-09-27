from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from rcwa_app.adapters.solver_mock.engine import MockSolverEngine
from rcwa_app.domain.models import ModelConfig, SweepRequest
from rcwa_app.orchestration.session import default_config
from rcwa_app.plotting_plotly.presenter import PlotPresenterPlotly


@pytest.fixture(scope="session")
def small_grids() -> Tuple[List[float], List[float]]:
    """Small λ and θ grids for deterministic tests."""
    return [3.0, 4.0, 5.0], [0.0, 15.0, 30.0]


@pytest.fixture(scope="session")
def presenter() -> PlotPresenterPlotly:
    """Plotly presenter under test."""
    return PlotPresenterPlotly()


@pytest.fixture(scope="session")
def small_result(small_grids: Tuple[List[float], List[float]]) -> Any:
    """Run the mock engine on a tiny grid and return its result object."""
    lam, th = small_grids
    cfg: ModelConfig = default_config()
    req = SweepRequest(config=cfg, sweep_lambda=True, lambda_grid_um=lam, theta_grid_deg=th)
    engine = MockSolverEngine()
    return engine.run(req)


@pytest.fixture(scope="session")
def spectral_fig(small_result: Any, presenter: PlotPresenterPlotly) -> Any:
    """Single-θ spectral plot at θ=15°."""
    return presenter.spectral_plot(small_result, fixed_theta=15.0)


@pytest.fixture(scope="session")
def map_fig(small_result: Any, presenter: PlotPresenterPlotly) -> Any:
    """ε(λ,θ) map figure."""
    return presenter.map_eps(small_result)
