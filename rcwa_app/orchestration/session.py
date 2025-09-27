# """
# Session and reducers: a thin orchestration layer independent of UI.
# """

# ruff: noqa: D401
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Tuple

import numpy as np

# -------------------------
# Configuration data model
# -------------------------


@dataclass
class SurfaceConfig:
    """Two-sinusoid relief parameters (DLIP-like)."""

    Ax_um: float = 0.60
    Ay_um: float = 0.60
    Lx_um: float = 5.00
    Ly_um: float = 5.00
    phix_rad: float = 0.0
    phiy_rad: float = 0.0
    rot_deg: float = 0.0
    duty: float = 0.50  # <-- added: required by domain surface model


@dataclass
class GeometryConfig:
    surface: SurfaceConfig = SurfaceConfig()


Pol = Literal["TE", "TM", "UNPOL"]


@dataclass
class IlluminationConfig:
    """Wavelength/angle spans and polarization."""

    # (min, max, count)
    lambda_um: Tuple[float, float, int] = (3.0, 22.0, 121)
    theta_deg: Tuple[float, float, int] = (0.0, 80.0, 41)
    polarization: Pol = "TM"


@dataclass
class NumericsConfig:
    """Numerical knobs (mock engine ignores most)."""

    N_orders: int = 11
    tol: float = 1e-6
    factorization: Literal["LI_FAST", "LI_STRICT", "NONE"] = "LI_FAST"


@dataclass
class StackConfig:
    """Minimal stack parameters referenced by orchestration."""

    textured_thickness_um: float = 2.0


@dataclass
class AppConfig:
    geometry: GeometryConfig = GeometryConfig()
    illumination: IlluminationConfig = IlluminationConfig()
    numerics: NumericsConfig = NumericsConfig()
    stack: StackConfig = StackConfig()


@dataclass
class AppSession:
    """Top-level container passed between UI, orchestration, and engine."""

    config: AppConfig
    last_request: Mapping[str, Any] | None = None
    last_result: Any | None = None  # xarray-like result from the engine


# -------------------------
# Session lifecycle helpers
# -------------------------


def init_session() -> AppSession:
    """
    Create a fresh session with sensible defaults.
    """
    cfg = AppConfig()
    # If your domain layer expects additional constructor args (e.g., Layer
    # k_override/transparent_cap_depth_um), those belong in the engine/stack
    # composition—here we keep only UI-facing config.
    return AppSession(config=cfg, last_request=None, last_result=None)


def update_geometry(session: AppSession, **kwargs: Any) -> AppSession:
    """
    Update geometry.surface fields from keyword arguments.

    Accepts keys: {"Ax_um","Ay_um","Lx_um","Ly_um","phix_rad","phiy_rad","rot_deg","duty"}.
    Unknown keys are ignored (no error), keeping UI interactions robust.
    """
    allowed = {
        "Ax_um",
        "Ay_um",
        "Lx_um",
        "Ly_um",
        "phix_rad",
        "phiy_rad",
        "rot_deg",
        "duty",  # <-- ensure duty can be set from the UI
    }
    surf = session.config.geometry.surface
    for k, v in kwargs.items():
        if k in allowed:
            setattr(surf, k, float(v))
    return session


def update_illumination(
    session: AppSession,
    *,
    lambda_span: Tuple[float, float, int],
    theta_span: Tuple[float, float, int],
    polarization: Pol,
) -> AppSession:
    """
    Update illumination spans and polarization.
    """
    lam_min, lam_max, nlam = lambda_span
    th_min, th_max, nth = theta_span

    ill = session.config.illumination
    ill.lambda_um = (float(lam_min), float(lam_max), int(nlam))
    ill.theta_deg = (float(th_min), float(th_max), int(nth))
    ill.polarization = polarization
    return session


def build_sweep_request(
    session: AppSession,
    *,
    sweep_lambda: bool = True,
    sweep_theta: bool = True,
) -> Mapping[str, Any]:
    """
    Materialize the wavelength/angle grids and pack an engine request.

    The mock engine expects:
      - "lambda_um": 1D array
      - "theta_deg": 1D array
      - "polarization": {"TE","TM","UNPOL"}
      - "surface": dict of surface parameters (including 'duty')
      - "numerics": dict with 'N_orders' and 'tol'
      - optional extras are ignored by the mock
    """
    lam_min, lam_max, nlam = session.config.illumination.lambda_um
    th_min, th_max, nth = session.config.illumination.theta_deg

    lam = np.linspace(lam_min, lam_max, nlam) if sweep_lambda else np.array([lam_min])
    th = np.linspace(th_min, th_max, nth) if sweep_theta else np.array([th_min])

    surf = session.config.geometry.surface
    surface_payload = {
        "Ax_um": surf.Ax_um,
        "Ay_um": surf.Ay_um,
        "Lx_um": surf.Lx_um,
        "Ly_um": surf.Ly_um,
        "phix_rad": surf.phix_rad,
        "phiy_rad": surf.phiy_rad,
        "rot_deg": surf.rot_deg,
        "duty": surf.duty,  # <-- explicitly present
    }

    numerics_payload = {
        "N_orders": session.config.numerics.N_orders,
        "tol": session.config.numerics.tol,
        "factorization": session.config.numerics.factorization,
    }

    req = {
        "lambda_um": lam,
        "theta_deg": th,
        "polarization": session.config.illumination.polarization,
        "surface": surface_payload,
        "numerics": numerics_payload,
        "stack": {
            "textured_thickness_um": session.config.stack.textured_thickness_um,
            # domain-level fields like k_override/transparent_cap_depth_um
            # are engine concerns for real RCWA; the mock ignores them.
            "k_override": None,
            "transparent_cap_depth_um": 0.0,
        },
    }

    return req
