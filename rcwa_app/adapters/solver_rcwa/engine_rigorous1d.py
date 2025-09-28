from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np
import xarray as xr

from rcwa_app.adapters.solver_rcwa.engine import RcwaSolverEngine  # planar totals
from rcwa_app.adapters.solver_rcwa.rigorous1d import symmetric_orders
from rcwa_app.domain.models import ModelConfig, SolverScalars, SweepRequest


@dataclass
class Rcwa1DRigorousResult:
    scalars: SolverScalars
    data: xr.Dataset


class Rcwa1DRigorousEngine:
    """Rigorous RCWA 1D (skeleton): contracts preserved, totals from planar TMM.

    Next step: replace the internal propagation with Li/Lalanne factorization + S-matrix.
    """

    def __init__(self) -> None:
        self.planar = RcwaSolverEngine()

    def run(self, request: Mapping[str, Any] | SweepRequest) -> Rcwa1DRigorousResult:
        base = self.planar.run(request)
        ds0 = base.data

        lam = ds0["lambda_um"].values.astype(float)
        th = ds0["theta_deg"].values.astype(float)

        # Orders from config
        if isinstance(request, SweepRequest):
            cfg = request.config
        else:
            cfg = cast(ModelConfig, cast(Mapping[str, Any], request)["config"])

        m = symmetric_orders(int(getattr(cfg.numerics, "N_orders", 5)))

        # For now: rigorous engine emits only m=0, equal to totals (planar limit).
        # This keeps tests green while we wire the eigen-solver.
        Rsum = ds0["Rsum"].values
        Tsum = ds0["Tsum"].values
        Asum = ds0["Asum"].values
        eps = ds0["Asum"].values  # emissivity

        Rm = np.zeros((m.size, lam.size, th.size), dtype=float)
        Tm = np.zeros_like(Rm)
        idx0 = int(np.where(m == 0)[0][0])
        Rm[idx0, :, :] = Rsum
        Tm[idx0, :, :] = Tsum

        ds = xr.Dataset(
            data_vars=dict(
                eps=(("lambda_um", "theta_deg"), eps),
                Rsum=(("lambda_um", "theta_deg"), Rsum),
                Tsum=(("lambda_um", "theta_deg"), Tsum),
                Asum=(("lambda_um", "theta_deg"), Asum),
                Rm=(("order", "lambda_um", "theta_deg"), Rm),
                Tm=(("order", "lambda_um", "theta_deg"), Tm),
            ),
            coords=dict(lambda_um=lam, theta_deg=th, order=m),
            attrs=dict(ds0.attrs),
        )

        residual = float(np.nanmax(np.abs(ds["Rsum"] + ds["Tsum"] + ds["Asum"] - 1.0)))
        scalars = SolverScalars(energy_residual=residual, notes="rcwa-1d-rigorous-skeleton")
        return Rcwa1DRigorousResult(scalars=scalars, data=ds)
