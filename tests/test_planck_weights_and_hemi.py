from __future__ import annotations

import numpy as np
import xarray as xr

from rcwa_app.thermal.planck import (
    hemispherical_average_eps,
    planck_weight_lambda,
    planck_weighted_band_eps,
)


def test_planck_weight_lambda_normalizes_and_is_positive() -> None:
    lam = np.linspace(3.0, 12.0, 51)
    w = planck_weight_lambda(lam, T_K=1000.0)
    assert w.shape == lam.shape
    assert np.all(w >= 0.0)
    assert abs(float(w.sum()) - 1.0) < 1e-12


def _toy_ds() -> xr.Dataset:
    lam = np.linspace(3.0, 12.0, 21)
    th = np.linspace(0.0, 60.0, 7)
    # simple separable emissivity: eps(λ, θ) = f(λ) * cos^2 θ, clipped to [0,1]
    f = np.clip((lam - lam.min()) / (lam.max() - lam.min()), 0.0, 1.0)
    F = np.broadcast_to(f[:, None], (lam.size, th.size))
    TH = np.deg2rad(np.broadcast_to(th[None, :], (lam.size, th.size)))
    eps = np.clip(F * (np.cos(TH) ** 2), 0.0, 1.0)
    return xr.Dataset(
        data_vars=dict(eps=(("lambda_um", "theta_deg"), eps)),
        coords=dict(lambda_um=lam, theta_deg=th),
    )


def test_hemispherical_average_monotonic_in_lambda() -> None:
    ds = _toy_ds()
    hemi = hemispherical_average_eps(ds)
    # hemispherical average should increase with λ for this toy model
    assert hemi.ndim == 1 and hemi.sizes["lambda_um"] == ds.sizes["lambda_um"]
    diffs = np.diff(hemi.values)
    assert np.all(diffs >= -1e-12)


def test_planck_weighted_band_eps_returns_scalar_and_weights() -> None:
    ds = _toy_ds()
    val_hemi, w = planck_weighted_band_eps(ds, T_K=1200.0, theta_deg=None)
    assert 0.0 <= val_hemi <= 1.0
    assert w.shape == ds["lambda_um"].shape

    # At normal incidence, line should be ≥ hemispherical (since cos^2 factor ≤ 1)
    val_normal, _ = planck_weighted_band_eps(ds, T_K=1200.0, theta_deg=0.0)
    assert val_normal >= val_hemi - 1e-12
