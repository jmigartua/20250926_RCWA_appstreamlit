from __future__ import annotations

import inspect

from rcwa_app.adapters.registry import list_engines, make_engine


def test_registry_returns_callable_run() -> None:
    names = list_engines()
    assert "Mock" in names and "Planar TMM" in names and "RCWA-1D" in names
    for name in names:
        eng = make_engine(name)
        assert hasattr(eng, "run") and callable(getattr(eng, "run"))
        # signature is (request) -> result-like; we only assert it accepts 1 arg
        sig = inspect.signature(eng.run)
        assert len(sig.parameters) == 1
