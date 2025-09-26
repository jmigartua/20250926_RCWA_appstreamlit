#"""
#Mock solver engine implementing the SolverEngine port.
#Produces shape-correct, deterministic surrogate data satisfying contracts.md.
#"""
from __future__ import annotations
import numpy as np
import xarray as xr
from typing import List

from rcwa_app.domain.models import SweepRequest, SolverResult, SolverScalars
from rcwa_app.domain.ports import SolverEngine


class MockSolverEngine(SolverEngine):
    def run(self, req: SweepRequest) -> SolverResult:
        cfg = req.config
        lam = np.array(req.lambda_grid_um or [])
        th = np.array(req.theta_grid_deg or [])
        pols: List[str] = ["TE", "TM"]  # always provide both for UI convenience

        # Geometry-driven surrogate parameters
        Ax, Ay = cfg.geometry.surface.Ax_um, cfg.geometry.surface.Ay_um
        Lx, Ly = cfg.geometry.surface.Lx_um, cfg.geometry.surface.Ly_um
        rot = np.deg2rad(cfg.geometry.surface.rot_deg)

        base = 0.6 * (Lx + Ly)
        peak1 = max(0.2, base * (1.0 + 0.15 * np.sin(rot)))
        peak2 = max(0.2, 0.55 * base)
        width = 0.08 * base + 0.04 * (Ax + Ay) + 0.05
        amp_TM = np.tanh(0.5 + 0.8 * (Ax + Ay) / (Lx + Ly + 1e-6))
        amp_TE = 0.75 * amp_TM

        # Angle dependence
        cos_th = np.cos(np.deg2rad(th))
        ang_gain = (0.6 + 0.4 * cos_th**2)  # shape (theta,)

        # Build eps(lambda, theta, pol)
        lam2d, th2d = np.meshgrid(lam, th, indexing="ij")
        g = lambda mu, sig: np.exp(-0.5 * ((lam2d - mu) / (sig + 1e-9)) ** 2)
        base_profile = 0.6 * g(peak1, width) + 0.4 * g(peak2, 1.4 * width)
        base_profile *= (0.3 + 0.7 * np.clip((Ax + Ay) / (Lx + Ly + 1e-6), 0.0, 1.0))

        eps_TM = np.clip(amp_TM * base_profile * ang_gain[None, :], 0.0, 1.0)
        eps_TE = np.clip(amp_TE * base_profile * ang_gain[None, :], 0.0, 1.0)

        eps = np.stack([eps_TE, eps_TM], axis=-1)  # (lambda, theta, pol)

        # Radiative partitions
        Tsum = 0.02 * (1.0 - base_profile)  # small nonzero transmission
        Tsum = np.clip(Tsum, 0.0, 0.05)
        Tsum = np.stack([Tsum, Tsum], axis=-1)

        Asum = eps  # Kirchhoff equality in this surrogate
        Rsum = np.clip(1.0 - Asum - Tsum, 0.0, 1.0)

        # Order-resolved splitting across m ∈ [-2..2]
        orders = np.arange(-2, 3, 1)
        w = np.exp(-0.5 * (orders / 1.0) ** 2)
        w = w / w.sum()
        Rm = Rsum[..., None] * w  # broadcast to (λ, θ, pol, order)
        Tm = Tsum[..., None] * w
        # Reorder dims to (order, λ, θ, pol)
        Rm = np.moveaxis(Rm, -1, 0)
        Tm = np.moveaxis(Tm, -1, 0)

        ds = xr.Dataset(
            data_vars=dict(
                eps=(("lambda_um", "theta_deg", "pol"), eps),
                Rsum=(("lambda_um", "theta_deg", "pol"), Rsum),
                Tsum=(("lambda_um", "theta_deg", "pol"), Tsum),
                Asum=(("lambda_um", "theta_deg", "pol"), Asum),
                Rm=(("order", "lambda_um", "theta_deg", "pol"), Rm),
                Tm=(("order", "lambda_um", "theta_deg", "pol"), Tm),
            ),
            coords=dict(
                lambda_um=lam,
                theta_deg=th,
                pol=("pol", pols),
                order=("order", orders),
            ),
            attrs=dict(
                notes="Mock surrogate; satisfies contracts.md; not physically accurate.",
            ),
        )

        # Energy residual diagnostic
        residual = float(np.nanmax(np.abs(ds["Rsum"].values + ds["Tsum"].values + ds["Asum"].values - 1.0)))
        scalars = SolverScalars(energy_residual=residual, notes="mock")

        return SolverResult(data=ds, scalars=scalars, schema_version="1.0.0")