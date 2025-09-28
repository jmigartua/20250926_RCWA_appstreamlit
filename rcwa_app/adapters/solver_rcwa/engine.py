from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence, cast

import numpy as np
import xarray as xr

from rcwa_app.adapters.materials.builtin import MaterialDB
from rcwa_app.adapters.solver_rcwa.tmm import LayerSpec, Pol, rt_to_RT, tmm_rt
from rcwa_app.domain.models import ModelConfig, SolverScalars, SweepRequest


@dataclass
class RcwaResult:
    scalars: SolverScalars
    data: xr.Dataset


class RcwaSolverEngine:
    """Planar TMM baseline (RCWA-ready API). Order m=0 only; contracts preserved.

    Future work adds Fourier factorization and true grating orders. Public outputs stay the same.
    """

    def __init__(self) -> None:
        self.mat = MaterialDB()

    # --- helpers -------------------------------------------------------------
    def _layers_from_config(self, cfg: ModelConfig) -> tuple[complex, complex, list[LayerSpec]]:
        # ambient (superstrate) and substrate; default to Air
        n0 = self.mat.n_of_lambda("Air", 1.0)
        ns = self.mat.n_of_lambda("Air", 1.0)
        finite: list[LayerSpec] = []
        for L in cfg.geometry.stack:
            nL = self.mat.n_of_lambda(L.material_ref, float(cfg.illumination.lambda_um[0]))
            if L.thickness_um is None:
                ns = nL
            else:
                finite.append(LayerSpec(n=nL, d_um=float(L.thickness_um)))
        return n0, ns, finite

    # --- API -----------------------------------------------------------------
    def run(self, request: Mapping[str, Any] | SweepRequest) -> RcwaResult:
        # Accept either a Pydantic request or a dict-like; prefer typed
        if isinstance(request, SweepRequest):
            cfg = request.config
            lam = np.asarray(request.lambda_grid_um, dtype=float)
            th = np.asarray(request.theta_grid_deg, dtype=float)
        else:
            d = cast(Mapping[str, Any], request)
            cfg = cast(ModelConfig, d.get("config"))
            lam = np.asarray(d["lambda_grid_um"], dtype=float)
            th = np.asarray(d["theta_grid_deg"], dtype=float)

        n0, ns, finite = self._layers_from_config(cfg)

        # Allocate arrays on (λ, θ)
        L, T = lam.size, th.size
        Rsum = np.zeros((L, T), dtype=float)
        Tsum = np.zeros((L, T), dtype=float)
        Asum = np.zeros((L, T), dtype=float)

        # Polarizations: UNPOL = average of TE/TM
        pols: Sequence[str] = (
            (cfg.illumination.polarization,)
            if cfg.illumination.polarization in ("TE", "TM")
            else ("TE", "TM")
        )

        for i, wl in enumerate(lam):
            for j, ang in enumerate(th):
                R_acc = 0.0
                T_acc = 0.0
                for pol in pols:
                    p = cast(Pol, pol)
                    r, t = tmm_rt(p, n0, ns, finite, float(wl), float(ang))
                    R, Tpow = rt_to_RT(p, n0, ns, r, t)
                    R_acc += R
                    T_acc += Tpow
                R_acc /= float(len(pols))
                T_acc /= float(len(pols))
                Rsum[i, j] = R_acc
                Tsum[i, j] = T_acc
                Asum[i, j] = max(0.0, 1.0 - R_acc - T_acc)

        # Emissivity ε = Asum (Kirchhoff)
        eps = Asum.copy()

        # Only order m=0 available in planar baseline
        orders = np.array([0], dtype=int)
        Rm = Rsum[None, :, :]
        Tm = Tsum[None, :, :]

        ds = xr.Dataset(
            data_vars=dict(
                eps=(("lambda_um", "theta_deg"), eps),
                Rsum=(("lambda_um", "theta_deg"), Rsum),
                Tsum=(("lambda_um", "theta_deg"), Tsum),
                Asum=(("lambda_um", "theta_deg"), Asum),
                Rm=(("order", "lambda_um", "theta_deg"), Rm),
                Tm=(("order", "lambda_um", "theta_deg"), Tm),
            ),
            coords=dict(lambda_um=lam, theta_deg=th, order=orders),
            attrs=dict(
                polarization=cfg.illumination.polarization, note="RCWA engine (planar TMM baseline)"
            ),
        )

        residual = float(np.nanmax(np.abs(Rsum + Tsum + Asum - 1.0)))
        scalars = SolverScalars(energy_residual=residual, notes="rcwa-baseline")
        return RcwaResult(scalars=scalars, data=ds)  # same shape/signature pattern as Mock
