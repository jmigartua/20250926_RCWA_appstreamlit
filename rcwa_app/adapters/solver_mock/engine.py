# """
# Mock solver engine implementing the SolverEngine port.
# Produces shape-correct, deterministic surrogate data satisfying contracts.md.
# """
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Union

import numpy as np
import xarray as xr
from numpy.typing import NDArray

# -------------------------
# Small typed helpers
# -------------------------


def g(
    mu: Union[float, NDArray[np.floating]], sig: Union[float, NDArray[np.floating]]
) -> Union[float, NDArray[np.floating]]:
    """
    Simple Gaussian envelope used by the mock to shape emissivity.
    """
    return np.exp(-0.5 * (mu / np.clip(sig, 1e-12, None)) ** 2)


@dataclass(frozen=True)
class _Scalars:
    energy_residual: float


@dataclass(frozen=True)
class MockResult:
    """
    Minimal result object consumed by the Plotly presenter and the UI.
    """

    data: xr.Dataset
    scalars: _Scalars


# -------------------------
# Mock solver engine
# -------------------------


class MockSolverEngine:
    """
    Deterministic, fast mock that synthesizes an (λ, θ) emissivity map.

    It preserves the public shape expected by the UI:
      - result.data: xarray Dataset with coords ("lambda_um","theta_deg")
      - variables: emissivity ε(λ, θ) in [0, 1], plus synthetic R,T
      - result.scalars.energy_residual: small non-negative number
    """

    def run(self, request: Mapping[str, Any]) -> MockResult:
        lam: NDArray[np.floating] = np.asarray(request["lambda_um"], dtype=float)
        th: NDArray[np.floating] = np.asarray(request["theta_deg"], dtype=float)
        pol: str = str(request.get("polarization", "TM"))

        # Use surface parameters to bias the synthetic ε
        surf: Mapping[str, Any] = request.get("surface", {})
        Ax = float(surf.get("Ax_um", 0.6))
        Ay = float(surf.get("Ay_um", 0.6))
        Lx = float(surf.get("Lx_um", 5.0))
        Ly = float(surf.get("Ly_um", 5.0))
        duty = float(surf.get("duty", 0.5))

        L = np.sqrt(Lx * Ly)
        lam0 = 0.6 * L + 0.1 * (Ax + Ay)
        th0 = 15.0 + 20.0 * (duty - 0.5)

        # Mesh
        Lam, Th = np.meshgrid(lam, th, indexing="ij")

        # Polarization knob for variety
        pol_factor = {"TM": 1.00, "TE": 0.85, "UNPOL": 0.925}.get(pol.upper(), 0.95)

        # Synthetic emissivity: smooth, bounded, with a ridge vs θ and a bump vs λ
        eps = (
            0.15
            + 0.80
            * g((Lam - lam0) / (0.15 * L + 1.0), 1.0)
            * g((Th - th0) / 25.0, 1.0)
            * pol_factor
        )
        eps = np.clip(eps, 0.0, 1.0)

        # Energy accounting (mock): A=ε, set R,T to sum close to 1
        A = eps
        R = 0.5 * (1.0 - A) * (1.0 + 0.2 * np.cos(2 * np.pi * Lam / (L + 1.0)))
        T = 1.0 - A - R
        residual = float(np.maximum(0.0, np.abs(1.0 - (R + T + A)).max()))

        ds = xr.Dataset(
            data_vars=dict(
                emissivity=(("lambda_um", "theta_deg"), A),
                R=(("lambda_um", "theta_deg"), R),
                T=(("lambda_um", "theta_deg"), T),
            ),
            coords=dict(lambda_um=lam, theta_deg=th),
            attrs=dict(polarization=pol),
        )

        return MockResult(data=ds, scalars=_Scalars(energy_residual=residual))
