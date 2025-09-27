# """
# Session and reducers: a thin orchestration layer independent of UI.
# """

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from rcwa_app.domain.models import (
    GeometryConfig,
    IlluminationConfig,
    Layer,
    ModelConfig,
    NumericsConfig,
    SolverResult,
    SweepRequest,
    ThermalPostConfig,
    TwoSinusoidSurface,
)


@dataclass(frozen=True)
class AppSession:
    config: ModelConfig
    last_request: SweepRequest | None = None
    last_result: SolverResult | None = None


def default_config() -> ModelConfig:
    geom = GeometryConfig(
        surface=TwoSinusoidSurface(
            Ax_um=0.2, Ay_um=0.2, Lx_um=6.0, Ly_um=6.0, phix_rad=0.0, phiy_rad=0.0, rot_deg=0.0
        ),
        stack=[
            Layer(name="Textured", thickness_um=2.0, material_ref="Steel"),
            Layer(name="Substrate", thickness_um=None, material_ref="Steel"),
        ],
    )
    ill = IlluminationConfig(
        lambda_um=(3.0, 22.0, 121), theta_deg=(0.0, 60.0, 61), psi_deg=0.0, polarization="TM"
    )
    num = NumericsConfig(N_orders=15, factorization="LI_FAST", s_matrix=True, tol=1e-6)
    cfg = ModelConfig(
        geometry=geom,
        illumination=ill,
        numerics=num,
        thermal=ThermalPostConfig(T_K=None, hemispherical=False, bands_um=[]),
        materials_model="tabulated",
        version="1.0.0",
    )
    return cfg


def init_session() -> AppSession:
    return AppSession(config=default_config())


def update_geometry(session: AppSession, **kwargs) -> AppSession:
    geom = session.config.geometry.model_copy(deep=True)
    surf = geom.surface.model_copy(
        update={k: v for k, v in kwargs.items() if hasattr(geom.surface, k)}
    )
    geom = geom.model_copy(update={"surface": surf})
    cfg = session.config.model_copy(update={"geometry": geom})
    return replace(session, config=cfg)


def update_illumination(
    session: AppSession,
    *,
    lambda_span: tuple[float, float, int] | None = None,
    theta_span: tuple[float, float, int] | None = None,
    polarization: str | None = None,
) -> AppSession:
    ill = session.config.illumination.model_copy(deep=True)
    if lambda_span is not None:
        ill.lambda_um = lambda_span
    if theta_span is not None:
        ill.theta_deg = theta_span
    if polarization is not None:
        ill.polarization = polarization  # type: ignore
    cfg = session.config.model_copy(update={"illumination": ill})
    return replace(session, config=cfg)


def build_sweep_request(session: AppSession, sweep_lambda: bool = True) -> SweepRequest:
    ill = session.config.illumination
    lam_min, lam_max, nlam = ill.lambda_um
    th_min, th_max, nth = ill.theta_deg
    lam = np.linspace(lam_min, lam_max, nlam).tolist()
    th = np.linspace(th_min, th_max, nth).tolist()
    req = SweepRequest(
        config=session.config,
        sweep_lambda=sweep_lambda,
        lambda_grid_um=lam,
        theta_grid_deg=th,
    )
    return req
