import importlib

import pytest


def test_domain_public_classes_exist() -> None:
    m = importlib.import_module("rcwa_app.domain.models")
    for name in (
        "ModelConfig",
        "GeometryConfig",
        "IlluminationConfig",
        "NumericsConfig",
        "SweepRequest",
        "SolverResult",
    ):
        assert hasattr(m, name), f"Missing domain symbol: {name}"


def test_mock_engine_exposes_run() -> None:
    eng_mod = importlib.import_module("rcwa_app.adapters.solver_mock.engine")
    Engine = getattr(eng_mod, "MockSolverEngine", None) or getattr(eng_mod, "Engine", None)
    assert Engine is not None and isinstance(Engine, type)
    engine = Engine()
    assert hasattr(engine, "run") and callable(engine.run)


def test_presenter_module_is_importable() -> None:
    pytest.importorskip("rcwa_app.plotting_plotly.presenter")
