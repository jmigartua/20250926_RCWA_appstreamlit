# rcwa_app/adapters/registry.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Type

# Only import the Protocol for typing (prevents circular imports at runtime)
if TYPE_CHECKING:
    # Typing-only to avoid runtime cycles; mypy sees the Protocol.
    from rcwa_app.domain.ports import SolverEngine  # pragma: no cover

# Concrete engines (adjust module paths if yours differ)
from rcwa_app.adapters.solver_mock.engine import MockSolverEngine
from rcwa_app.adapters.solver_rcwa.engine import RcwaSolverEngine  # planar TMM

# If your 1D grating engine class lives in grating1d.py, import from there instead:
# from rcwa_app.adapters.solver_rcwa.grating1d import Rcwa1DGratingEngine
from rcwa_app.adapters.solver_rcwa.engine_1d import Rcwa1DGratingEngine  # if this is your path
from rcwa_app.adapters.solver_rcwa.engine_rigorous1d import Rcwa1DRigorousEngine

__all__ = ["list_engines", "make_engine"]

# Registry: human-readable name → engine class
_REGISTRY: Dict[str, Type[Any]] = {
    "Mock": MockSolverEngine,
    "Planar TMM": RcwaSolverEngine,
    "RCWA-1D": Rcwa1DGratingEngine,
    "RCWA-1D (rigorous)": Rcwa1DRigorousEngine,
}


def list_engines() -> List[str]:
    return list(_REGISTRY.keys())


def make_engine(name: str, **kwargs: Any) -> SolverEngine:
    """
    Instantiate the requested engine. Extra kwargs are forwarded if the class
    accepts them (e.g., Rcwa1DRigorousEngine(mode="rigorous-lite")).
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise KeyError(f"Unknown engine '{name}'. Available: {', '.join(_REGISTRY)}")
    try:
        return cls(**kwargs)
    except TypeError:
        # Fallback if kwargs are not supported by a given engine
        if kwargs:
            return cls()
        raise
