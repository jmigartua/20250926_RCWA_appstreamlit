from __future__ import annotations
import pytest
from pydantic import ValidationError

from rcwa_app.orchestration.session import default_config
from rcwa_app.domain.models import (
    TwoSinusoidSurface,
    GeometryConfig,
    Layer,
    ModelConfig,
    IlluminationConfig,
    NumericsConfig,
)


def test_default_config_is_valid():
    cfg = default_config()
    # Basic sanity: geometry & illumination present, positive periods
    assert cfg.geometry.surface.Lx_um > 0
    assert cfg.geometry.surface.Ly_um > 0
    lam_min, lam_max, nlam = cfg.illumination.lambda_um
    th_min, th_max, nth = cfg.illumination.theta_deg
    assert lam_min < lam_max and nlam >= 3
    assert th_min <= th_max and nth >= 3


def test_invalid_geometry_raises():
    # Negative amplitude must be rejected by Pydantic validators
    with pytest.raises(ValidationError):
        _ = TwoSinusoidSurface(Ax_um=-0.1, Ay_um=0.2, Lx_um=6.0, Ly_um=6.0)


def test_model_roundtrip_update():
    cfg = default_config()
    new_geom = cfg.geometry.model_copy(update={
        "surface": cfg.geometry.surface.model_copy(update={"Ax_um": 0.5})
    })
    new_cfg = cfg.model_copy(update={"geometry": new_geom})
    assert new_cfg.geometry.surface.Ax_um == 0.5
    # Original object unchanged (immutability by copy)
    assert cfg.geometry.surface.Ax_um != 0.5