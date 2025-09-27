# """
# Mock solver engine implementing the SolverEngine port.
# Produces shape-correct, deterministic surrogate data satisfying contracts.md.
# """
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray


def g(mu: float, sig: float) -> float:
    """Gaussian helper used in the mock spectrum."""
    return float(np.exp(-(mu**2) / (2.0 * sig**2)))


@dataclass
class MockResult:
    scalars: Any
    data: xr.Dataset


def _grid_from_span(span: Any) -> NDArray[np.floating]:
    """Utility: (min, max, count) → np.linspace."""
    a, b, n = span
    return np.linspace(float(a), float(b), int(n))


def _normalize_request(request: Any) -> MutableMapping[str, Any]:
    """
    Normalize incoming request to a dict with keys the mock expects:
      {"lambda_um", "theta_deg", "polarization", "surface", "numerics", "stack"}.

    Accepts:
      • Mapping[str, Any]  (dict-like)
      • Pydantic v2 model  (.model_dump())
      • Plain object with attributes: config, lambda_grid_um, theta_grid_deg
    """
    # Case 1: already mapping
    if isinstance(request, Mapping):
        return dict(request)

    # Case 2: Pydantic v2 model: use .model_dump()
    dump = getattr(request, "model_dump", None)
    if callable(dump):
        dumped = dump()
        if isinstance(dumped, Mapping) and "config" in dumped:
            cfg = dumped["config"]
            out: dict[str, Any] = {}

            # Prefer explicit grids if present; otherwise derive from spans
            out["lambda_um"] = dumped.get("lambda_grid_um") or _grid_from_span(
                cfg["illumination"]["lambda_um"]
            )
            out["theta_deg"] = dumped.get("theta_grid_deg") or _grid_from_span(
                cfg["illumination"]["theta_deg"]
            )
            out["polarization"] = cfg["illumination"]["polarization"]

            # Surface & numerics from serialized config
            out["surface"] = cfg["geometry"]["surface"]
            out["numerics"] = cfg["numerics"]

            # Minimal stack payload for the mock
            out["stack"] = {
                "textured_thickness_um": 2.0,
                "k_override": None,
                "transparent_cap_depth_um": 0.0,
            }
            return out
        return dict(dumped)

    # Case 3: plain object with attributes (config + optional grids)
    out2: dict[str, Any] = {}

    if hasattr(request, "lambda_grid_um"):
        out2["lambda_um"] = cast(Any, getattr(request, "lambda_grid_um"))
    if hasattr(request, "theta_grid_deg"):
        out2["theta_deg"] = cast(Any, getattr(request, "theta_grid_deg"))

    cfg = getattr(request, "config", None)
    if cfg is not None:
        ill = getattr(cfg, "illumination", None)
        if ill is not None:
            out2.setdefault("polarization", getattr(ill, "polarization", "TM"))
            if "lambda_um" not in out2 and hasattr(ill, "lambda_um"):
                out2["lambda_um"] = _grid_from_span(getattr(ill, "lambda_um"))
            if "theta_deg" not in out2 and hasattr(ill, "theta_deg"):
                out2["theta_deg"] = _grid_from_span(getattr(ill, "theta_deg"))

        geom = getattr(cfg, "geometry", None)
        surf = getattr(geom, "surface", None) if geom is not None else None
        if surf is not None:
            out2["surface"] = {
                "Ax_um": float(getattr(surf, "Ax_um", 0.6)),
                "Ay_um": float(getattr(surf, "Ay_um", 0.6)),
                "Lx_um": float(getattr(surf, "Lx_um", 5.0)),
                "Ly_um": float(getattr(surf, "Ly_um", 5.0)),
                "phix_rad": float(getattr(surf, "phix_rad", 0.0)),
                "phiy_rad": float(getattr(surf, "phiy_rad", 0.0)),
                "rot_deg": float(getattr(surf, "rot_deg", 0.0)),
                "duty": float(getattr(surf, "duty", 0.5)),
            }

        num = getattr(cfg, "numerics", None)
        if num is not None:
            out2["numerics"] = {
                "N_orders": int(getattr(num, "N_orders", 11)),
                "tol": float(getattr(num, "tol", 1e-6)),
                "factorization": cast(str, getattr(num, "factorization", "LI_FAST")),
            }

    out2.setdefault(
        "stack",
        {"textured_thickness_um": 2.0, "k_override": None, "transparent_cap_depth_um": 0.0},
    )
    return out2


class MockSolverEngine:
    def run(self, request: Mapping[str, Any] | Any) -> MockResult:
        # Normalize any incoming representation to a plain dict
        req = _normalize_request(request)

        lam: NDArray[np.floating] = np.asarray(req["lambda_um"], dtype=float)
        th: NDArray[np.floating] = np.asarray(req["theta_deg"], dtype=float)
        pol: str = cast(str, req.get("polarization", "TM"))

        # Smooth emissivity field over (λ, θ)
        lam_grid, th_grid = np.meshgrid(lam, th, indexing="ij")

        surf = req.get("surface", {})
        ax = float(surf.get("Ax_um", 0.6))
        ay = float(surf.get("Ay_um", 0.6))
        duty = float(surf.get("duty", 0.5))

        sig_lam = 0.15 * (1.0 + ax + ay)
        sig_th = 10.0 * (1.0 + duty)

        base = np.exp(-((lam_grid - lam_grid.mean()) ** 2) / (2.0 * sig_lam**2))
        ang = np.exp(-((th_grid - th_grid.mean()) ** 2) / (2.0 * sig_th**2))
        pol_scale = 1.0 if pol == "TM" else (0.9 if pol == "TE" else 0.95)

        emiss = np.clip(pol_scale * base * ang, 0.0, 1.0)

        ds = xr.Dataset(
            data_vars=dict(
                eps=(("lambda_um", "theta_deg"), emiss),  # <-- was 'emissivity'
            ),
            coords=dict(
                lambda_um=lam,
                theta_deg=th,
            ),
            attrs=dict(
                polarization=pol,
                note="Mock engine result",
            ),
        )

        class _Scalars:
            energy_residual = float(np.abs(1.0 - emiss.mean()))

        return MockResult(scalars=_Scalars(), data=ds)
