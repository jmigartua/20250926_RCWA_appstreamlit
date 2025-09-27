from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Tuple

import numpy as np

# Domain (Pydantic v2) models used by tests and the app
from rcwa_app.domain.models import (
    GeometryConfig,
    IlluminationConfig,
    ModelConfig,
    NumericsConfig,
    TwoSinusoidSurface,
)

__all__ = [
    "AppSession",
    "default_config",
    "init_session",
    "update_geometry",
    "update_illumination",
    "build_sweep_request",
]

Pol = Literal["TE", "TM", "UNPOL"]


@dataclass
class AppSession:
    """
    Thin runtime container passed between UI, orchestration, and engine.

    `config` is the domain Pydantic model (ModelConfig), so consumers that validate
    against ModelConfig (e.g., SweepRequest(config=...)) accept it.
    """

    config: ModelConfig
    last_request: Mapping[str, Any] | None = None
    last_result: Any | None = None  # xarray-like result from the engine


# -------------------------
# Session lifecycle helpers
# -------------------------


def default_config() -> ModelConfig:
    """
    Return a fully-populated domain configuration (ModelConfig).

    Notes:
    • GeometryConfig.stack is a List[...] in your domain; provide an empty list.
    • Surface includes required `duty`.
    """
    surface = TwoSinusoidSurface(
        Ax_um=0.60,
        Ay_um=0.60,
        Lx_um=5.00,
        Ly_um=5.00,
        phix_rad=0.0,
        phiy_rad=0.0,
        rot_deg=0.0,
        duty=0.50,
    )

    geometry = GeometryConfig(
        surface=surface,
        stack=[],  # domain schema expects a list
    )

    illumination = IlluminationConfig(
        lambda_um=(3.0, 22.0, 121),
        theta_deg=(0.0, 80.0, 41),
        polarization="TM",
    )

    numerics = NumericsConfig(
        N_orders=11,
        tol=1e-6,
        factorization="LI_FAST",
    )

    return ModelConfig(
        geometry=geometry,
        illumination=illumination,
        numerics=numerics,
    )


def init_session() -> AppSession:
    """Create a fresh session with sensible defaults."""
    return AppSession(config=default_config(), last_request=None, last_result=None)


def update_geometry(session: AppSession, **kwargs: Any) -> AppSession:
    """
    Update geometry.surface fields from keyword arguments.

    Allowed keys:
      {"Ax_um","Ay_um","Lx_um","Ly_um","phix_rad","phiy_rad","rot_deg","duty"}.
    Unknown keys are ignored to keep UI interactions robust.
    """
    allowed = {
        "Ax_um",
        "Ay_um",
        "Lx_um",
        "Ly_um",
        "phix_rad",
        "phiy_rad",
        "rot_deg",
        "duty",
    }
    updates: dict[str, float] = {k: float(v) for k, v in kwargs.items() if k in allowed}
    if updates:
        new_surface = session.config.geometry.surface.model_copy(update=updates)
        new_geom = session.config.geometry.model_copy(update={"surface": new_surface})
        session.config = session.config.model_copy(update={"geometry": new_geom})
    return session


def update_illumination(
    session: AppSession,
    *,
    lambda_span: Tuple[float, float, int],
    theta_span: Tuple[float, float, int],
    polarization: Pol,
) -> AppSession:
    """Update illumination spans and polarization."""
    lam_min, lam_max, nlam = lambda_span
    th_min, th_max, nth = theta_span

    new_ill = session.config.illumination.model_copy(
        update={
            "lambda_um": (float(lam_min), float(lam_max), int(nlam)),
            "theta_deg": (float(th_min), float(th_max), int(nth)),
            "polarization": polarization,
        }
    )
    session.config = session.config.model_copy(update={"illumination": new_ill})
    return session


def build_sweep_request(
    session: AppSession,
    *,
    sweep_lambda: bool = True,
    sweep_theta: bool = True,
) -> Mapping[str, Any]:
    """
    Materialize the wavelength/angle grids and pack an engine request.

    Expected by the mock engine:
      - "lambda_um": 1D array
      - "theta_deg": 1D array
      - "polarization": {"TE","TM","UNPOL"}
      - "surface": dict of surface parameters (including 'duty')
      - "numerics": dict with 'N_orders' and 'tol'
      - "stack": minimal dict (the mock engine ignores geometry.stack, which is a list)
    """
    lam_min, lam_max, nlam = session.config.illumination.lambda_um
    th_min, th_max, nth = session.config.illumination.theta_deg

    lam = np.linspace(lam_min, lam_max, nlam) if sweep_lambda else np.array([lam_min], dtype=float)
    th = np.linspace(th_min, th_max, nth) if sweep_theta else np.array([th_min], dtype=float)

    s = session.config.geometry.surface
    surface_payload = {
        "Ax_um": s.Ax_um,
        "Ay_um": s.Ay_um,
        "Lx_um": s.Lx_um,
        "Ly_um": s.Ly_um,
        "phix_rad": s.phix_rad,
        "phiy_rad": s.phiy_rad,
        "rot_deg": s.rot_deg,
        "duty": s.duty,
    }

    numerics_payload = session.config.numerics.model_dump()

    # Provide a small synthetic stack payload for the mock engine.
    stack_payload = {
        "textured_thickness_um": 2.0,
        "k_override": None,
        "transparent_cap_depth_um": 0.0,
    }

    return {
        "lambda_um": lam,
        "theta_deg": th,
        "polarization": session.config.illumination.polarization,
        "surface": surface_payload,
        "numerics": numerics_payload,
        "stack": stack_payload,
    }
