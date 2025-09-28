from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np
import xarray as xr

from rcwa_app.adapters.solver_rcwa.engine import RcwaSolverEngine  # planar totals
from rcwa_app.adapters.solver_rcwa.grating1d import distribute_orders
from rcwa_app.domain.models import ModelConfig, SolverScalars, SweepRequest


@dataclass
class Rcwa1DResult:
    scalars: SolverScalars
    data: xr.Dataset


class Rcwa1DGratingEngine:
    """1D grating engine layered on top of planar totals.

    Uses a stable order-weight model consistent with the grating equation to distribute R/T into m-orders.
    Planar limit (amplitude→0) collapses to m=0.
    """

    def __init__(self) -> None:
        self.planar = RcwaSolverEngine()

    def run(self, request: Mapping[str, Any] | SweepRequest) -> Rcwa1DResult:
        # First get totals from planar engine to ensure energy consistency
        base = self.planar.run(request)
        ds0 = base.data

        # Extract grids
        lam = ds0["lambda_um"].values.astype(float)
        th = ds0["theta_deg"].values.astype(float)

        # Geometry (1D): period Λx and amplitude Ax from config
        if isinstance(request, SweepRequest):
            cfg = request.config
        else:
            cfg = cast(ModelConfig, cast(Mapping[str, Any], request)["config"])
        surf = cfg.geometry.surface
        period = float(getattr(surf, "Lx_um", 10.0))
        amplitude = float(getattr(surf, "Ax_um", 0.0))

        # Orders from numerics (symmetric set)
        N = int(getattr(cfg.numerics, "N_orders", 5))
        m_orders = np.arange(-N // 2, N // 2 + 1, dtype=int)
        if m_orders.size % 2 == 0:
            # ensure odd count so m=0 exists
            m_orders = np.arange(-N, N + 1, dtype=int)

        # Allocate
        Rm = np.zeros((m_orders.size, lam.size, th.size), dtype=float)
        Tm = np.zeros_like(Rm)

        # Distribute orders per (λ, θ)
        for i, wl in enumerate(lam):
            for j, ang in enumerate(th):
                Rsum = float(ds0["Rsum"].values[i, j])
                Tsum = float(ds0["Tsum"].values[i, j])
                rrow, trow = distribute_orders(
                    Rsum,
                    Tsum,
                    lambda_um=float(wl),
                    theta_deg=float(ang),
                    period_um=period,
                    amplitude_um=amplitude,
                    m_orders=m_orders,
                    n_ambient=1.0,
                    n_substrate=1.0,
                    pol="UNPOL",
                )
                Rm[:, i, j] = rrow
                Tm[:, i, j] = trow

        # Assemble dataset: copy totals from planar, replace Rm/Tm and keep totals consistent
        ds = xr.Dataset(
            data_vars=dict(
                eps=ds0["eps"],
                Rsum=ds0["Rsum"],
                Tsum=ds0["Tsum"],
                Asum=ds0["Asum"],
                Rm=(("order", "lambda_um", "theta_deg"), Rm),
                Tm=(("order", "lambda_um", "theta_deg"), Tm),
            ),
            coords=dict(lambda_um=lam, theta_deg=th, order=m_orders),
            attrs=dict(ds0.attrs),
        )
        residual = float(np.nanmax(np.abs(ds["Rsum"] + ds["Tsum"] + ds["Asum"] - 1.0)))
        scalars = SolverScalars(energy_residual=residual, notes="rcwa-1d-weighted")
        return Rcwa1DResult(scalars=scalars, data=ds)
