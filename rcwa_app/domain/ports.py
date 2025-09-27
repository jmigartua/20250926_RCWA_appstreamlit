# """
# Ports (interfaces) for adapters. The UI and orchestration depend ONLY on these.
# """
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import SolverResult, SweepRequest


class MaterialDB(ABC):
    @abstractmethod
    def list_materials(self) -> list[str]:
        """Return available material identifiers."""

    @abstractmethod
    def get_nk(self, name: str, lambda_um: list[float]) -> Any:
        """Return xarray.Dataset with coords lambda_um and data vars n, k."""


class SolverEngine(ABC):
    @abstractmethod
    def run(self, req: SweepRequest) -> SolverResult:
        """Execute a sweep and return results conforming to contracts.md."""


class PlotPresenter(ABC):
    @abstractmethod
    def spectral_plot(self, result: SolverResult, fixed_theta: float) -> Any:
        """Figure: ε(λ) at closest θ; unpolarized average if multiple pols are present."""

    @abstractmethod
    def map_eps(self, result: SolverResult) -> Any:
        """Figure: heatmap of ε(λ, θ), unpolarized average if applicable."""

    @abstractmethod
    def orders_plot(self, result: SolverResult, i_lambda: int, i_theta: int) -> Any:
        """Figure: order-resolved Rm/Tm at a grid index (may be a placeholder)."""
