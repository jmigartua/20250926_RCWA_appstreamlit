from __future__ import annotations

from typing import Any

from rcwa_app.domain.models import ModelConfig, TwoSinusoidSurface
from rcwa_app.orchestration.session import default_config


def test_default_config_is_valid() -> None:
    """
    default_config() must return a fully-populated Pydantic v2 root (ModelConfig)
    with the expected top-level sections present.
    """
    cfg: ModelConfig = default_config()
    assert isinstance(cfg, ModelConfig)

    dumped: dict[str, Any] = cfg.model_dump()
    for key in ("geometry", "illumination", "numerics"):
        assert key in dumped, f"missing required section: {key}"


def test_model_roundtrip_update() -> None:
    """
    Model copies should be immutable by default and support .model_copy(update=...)
    semantics on nested models (geometry.surface in this case).
    """
    cfg: ModelConfig = default_config()

    # Update a nested field immutably via Pydantic's model_copy
    new_geom = cfg.geometry.model_copy(
        update={"surface": cfg.geometry.surface.model_copy(update={"Ax_um": 0.5})}
    )

    # New geometry reflects the change
    assert new_geom.surface.Ax_um == 0.5

    # Original remains unchanged
    assert cfg.geometry.surface.Ax_um != 0.5


def test_surface_requires_duty() -> None:
    """
    TwoSinusoidSurface requires the 'duty' parameter (as per domain model).
    Guard against Optional[float] in static typing.
    """
    surf = TwoSinusoidSurface(
        Ax_um=0.60,
        Ay_um=0.40,
        Lx_um=5.00,
        Ly_um=5.00,
        phix_rad=0.0,
        phiy_rad=0.0,
        rot_deg=0.0,
        duty=0.50,  # required by the model
    )
    # mypy: surf.duty is float | None, so guard before numeric comparisons
    assert surf.duty is not None
    duty: float = surf.duty
    assert 0.0 <= duty <= 1.0
