from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, cast

import numpy as np
import xarray as xr

from rcwa_app.adapters.solver_rcwa.coupling import propagating_mask
from rcwa_app.adapters.solver_rcwa.engine import RcwaSolverEngine  # planar totals
from rcwa_app.adapters.solver_rcwa.rigorous1d import kx_orders, symmetric_orders
from rcwa_app.domain.models import ModelConfig, SolverScalars, SweepRequest

Mode = Literal["skeleton", "rigorous-lite"]


@dataclass
class Rcwa1DRigorousResult:
    scalars: SolverScalars
    data: xr.Dataset


class Rcwa1DRigorousEngine:
    """
    Rigorous RCWA 1D (switchable):

    - mode="skeleton" (default): keep all power in m=0 (previous behavior).
    - mode="rigorous-lite": distribute Rsum/Tsum across propagating orders
      deterministically using a smooth, angle-dependent weight that respects
      the grating equation and preserves the totals exactly.

    This keeps UI/ports/tests stable while we wire the full Li/Lalanne core.
    """

    def __init__(self, mode: Mode = "skeleton") -> None:
        self.planar = RcwaSolverEngine()
        self.mode: Mode = mode

    def run(self, request: Mapping[str, Any] | SweepRequest) -> Rcwa1DRigorousResult:
        base = self.planar.run(request)
        ds0 = base.data

        lam = ds0["lambda_um"].values.astype(float)  # (L,)
        th = ds0["theta_deg"].values.astype(float)  # (T,)

        # Load config irrespective of representation
        if isinstance(request, SweepRequest):
            cfg = request.config
        else:
            cfg = cast(ModelConfig, cast(Mapping[str, Any], request)["config"])

        # Orders from numerics
        m = symmetric_orders(int(getattr(cfg.numerics, "N_orders", 5)))  # (M,)

        # Totals and emissivity from planar baseline
        Rsum = ds0["Rsum"].values  # (L, T)
        Tsum = ds0["Tsum"].values  # (L, T)
        Asum = ds0["Asum"].values  # (L, T)
        eps = ds0["Asum"].values  # emissivity = absorptance here

        # Allocate order-resolved arrays
        Rm = np.zeros((m.size, lam.size, th.size), dtype=float)
        Tm = np.zeros_like(Rm)

        if self.mode == "skeleton":
            # All power in m=0 (previous behavior)
            idx0 = int(np.where(m == 0)[0][0])
            Rm[idx0, :, :] = Rsum
            Tm[idx0, :, :] = Tsum
        else:
            # --- rigorous-lite: distribute across propagating orders ---
            # Period along x (1D grating); fallback to a sane default if not set.
            period_um = float(getattr(cfg.geometry.surface, "Lx_um", 10.0))
            n_ambient = 1.0  # ambient refractive index (conservative default)
            # For each (λ, θ), compute which orders propagate and center m0 from grating equation
            for j_th, theta in enumerate(th):
                for i_lam, wl in enumerate(lam):
                    k0 = 2.0 * np.pi / float(wl)
                    kx = kx_orders(
                        lambda_um=wl, theta_deg=theta, period_um=period_um, m=m, n_medium=n_ambient
                    )
                    mask = propagating_mask(k0, n_ambient, kx)
                    if not mask.any():
                        # no propagating orders: keep totals in m=0 for safety
                        idx0 = int(np.where(m == 0)[0][0])
                        Rm[idx0, i_lam, j_th] = float(Rsum[i_lam, j_th])
                        Tm[idx0, i_lam, j_th] = float(Tsum[i_lam, j_th])
                        continue

                    # Center order predicted by grating equation: m0 ≈ round(-n sinθ * Λ / λ)
                    m0 = int(np.round(-n_ambient * np.sin(np.deg2rad(theta)) * (period_um / wl)))
                    # Smooth Gaussian weights around m0, only for propagating orders
                    dm = (m - m0).astype(float)
                    # Width tied to angular bandwidth; keep finite in grazing angles
                    sigma = 1.25 + 0.35 * abs(np.sin(np.deg2rad(theta)))
                    w = np.exp(-0.5 * (dm / sigma) ** 2)
                    w = w * mask.astype(float)
                    s = float(w.sum())
                    if s <= 0.0:
                        # if numerical edge case, fall back to m=0
                        idx0 = int(np.where(m == 0)[0][0])
                        Rm[idx0, i_lam, j_th] = float(Rsum[i_lam, j_th])
                        Tm[idx0, i_lam, j_th] = float(Tsum[i_lam, j_th])
                    else:
                        w = w / s
                        Rm[:, i_lam, j_th] = w * float(Rsum[i_lam, j_th])
                        Tm[:, i_lam, j_th] = w * float(Tsum[i_lam, j_th])

        # Assemble Dataset
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
        scalars = SolverScalars(energy_residual=residual, notes=f"rcwa-1d-rigorous({self.mode})")
        return Rcwa1DRigorousResult(scalars=scalars, data=ds)
