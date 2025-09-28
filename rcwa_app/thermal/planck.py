from __future__ import annotations

from typing import Tuple

import numpy as np
import xarray as xr


def planck_weight_lambda(lambda_um: np.ndarray, T_K: float) -> np.ndarray:
    r"""
    Normalized Planck spectral weight over a *discrete* wavelength grid (μm).

    b_λ(λ, T) ∝ (1/λ^5) * 1 / (exp(c2/(λ T)) - 1), with c2 = h c / k_B.
    We normalize so that sum_i w_i = 1 on the *given* λ grid using trapezoidal
    quadrature in λ. Units cancel in the normalized ratio.

    Returns an array w (shape (L,)) s.t. w ≥ 0 and ∑ w = 1 (within FP tolerance).
    """
    lam = np.asarray(lambda_um, dtype=float)
    if lam.ndim != 1 or (lam <= 0.0).any():
        raise ValueError("lambda_um must be a 1D array with positive wavelengths")

    T = float(T_K)
    if T <= 0.0:
        raise ValueError("Temperature must be > 0 K")

    # Second radiation constant c2 = h c / k_B in μm·K
    c2_umK = 1.438776877e4  # μm·K (CODATA ~ exact enough for our purposes)

    x = c2_umK / (lam * T)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        b = (lam**-5) * (1.0 / (np.expm1(x)))  # ∝ B_λ; safe for large x via expm1
        b = np.clip(b, 0.0, np.finfo(float).max)

    # Normalize with trapezoidal rule ∫ b(λ) dλ ≈ Σ w_i
    integral_f = float(np.trapezoid(b, lam))
    if not np.isfinite(integral_f) or integral_f <= 0.0:
        # fallback to uniform weights on pathological inputs
        w = np.full_like(lam, 1.0 / lam.size)
    else:
        w = b / integral_f

    # Final renormalization on the discrete grid (sum to 1 within float tol)
    w = w / max(1e-300, float(w.sum()))
    return w


def hemispherical_average_eps(ds: xr.Dataset, var: str = "eps") -> xr.DataArray:
    r"""
    Cosine-weighted hemispherical average over θ of an angular emissivity map:

        ε̄_hemis(λ) = ( ∫_0^{π/2} ε(λ, θ) cosθ sinθ dθ ) / ( ∫_0^{π/2} cosθ sinθ dθ )
                    = 2 ∫_0^{π/2} ε(λ, θ) cosθ sinθ dθ

    Assumes ds[var] has dims ("lambda_um", "theta_deg") with θ in degrees.
    Returns a DataArray of shape (lambda_um,).
    """
    if var not in ds.data_vars:
        raise KeyError(f"Dataset missing variable {var!r}")
    if "lambda_um" not in ds.dims or "theta_deg" not in ds.dims:
        raise KeyError("Dataset must have dims ('lambda_um', 'theta_deg')")

    lam = ds["lambda_um"].values
    th_deg = ds["theta_deg"].values
    th_rad = np.deg2rad(th_deg)

    # weights for integral: cosθ sinθ dθ over θ ∈ [0, π/2]
    w_th = np.cos(th_rad) * np.sin(th_rad)
    # broadcast to (λ, θ)
    W = np.broadcast_to(w_th, (lam.size, th_deg.size))

    eps = ds[var].values
    num = np.asarray(np.trapezoid(eps * W, th_rad, axis=1), dtype=float)
    # Hemispherical average: 2 * ∫ ε cosθ sinθ dθ
    out = 2.0 * num

    return xr.DataArray(
        out,
        coords=dict(lambda_um=lam),
        dims=("lambda_um",),
        name=f"{var}_hemis",
        attrs=dict(note="cosθ-weighted hemispherical average over θ"),
    )


def planck_weighted_band_eps(
    ds: xr.Dataset,
    T_K: float,
    theta_deg: float | None = None,
    var: str = "eps",
) -> Tuple[float, np.ndarray]:
    r"""
    Planck-weighted, band-integrated emissivity on the current λ grid.

    If theta_deg is None ⇒ use **hemispherical** ε̄_hemis(λ);
    else use ε(λ, θ=theta_deg) line. In both cases, apply normalized Planck weights.

    Returns (scalar_eps_bar, weights) where weights is the normalized w(λ).
    """
    lam = ds["lambda_um"].values
    w = planck_weight_lambda(lam, T_K)

    if theta_deg is None:
        eps_line = hemispherical_average_eps(ds, var=var).values
    else:
        # pick nearest θ index
        j = int(np.argmin(np.abs(ds["theta_deg"].values - float(theta_deg))))
        eps_line = ds[var].isel(theta_deg=j).values

    eps_bar = float(np.sum(w * np.clip(eps_line, 0.0, 1.0)))
    return eps_bar, w
