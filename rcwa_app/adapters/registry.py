from __future__ import annotations

from typing import Any, Dict, Protocol, Type

# Engines
from rcwa_app.adapters.solver_mock.engine import MockSolverEngine
from rcwa_app.adapters.solver_rcwa.engine import RcwaSolverEngine
from rcwa_app.adapters.solver_rcwa.engine_1d import Rcwa1DGratingEngine
from rcwa_app.adapters.solver_rcwa.engine_rigorous1d import Rcwa1DRigorousEngine


class SolverEngine(Protocol):
    def run(self, request: Any) -> Any: ...


_REGISTRY: Dict[str, Type[SolverEngine]] = {
    "Mock": MockSolverEngine,
    "Planar TMM": RcwaSolverEngine,
    "RCWA-1D": Rcwa1DGratingEngine,
    "RCWA-1D (rigorous)": Rcwa1DRigorousEngine,  # NEW
}


def list_engines() -> list[str]:
    return list(_REGISTRY.keys())


def make_engine(name: str) -> SolverEngine:
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown engine '{name}'. Available: {', '.join(_REGISTRY)}")
    return cls()
