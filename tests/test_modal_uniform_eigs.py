from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import (
    kx_orders,
    kz_from_dispersion,
    modal_uniform_te_tm,
    symmetric_orders,
)


def test_uniform_modal_matches_dispersion_relation() -> None:
    # Setup: n=1.5 uniform medium, lambda=5um, theta=10deg, 1D orders
    lam = 5.0
    theta = 10.0
    n = 1.5
    eps0 = n * n
    k0 = 2.0 * np.pi / lam
    m = symmetric_orders(5)
    kx = kx_orders(lambda_um=lam, theta_deg=theta, period_um=10.0, m=m, n_medium=1.0)

    # Modal (uniform) vs analytic dispersion
    gamma, W = modal_uniform_te_tm(eps0=eps0, k0=k0, kx=kx)
    gamma_ref = kz_from_dispersion(k0, n, kx)

    assert gamma.shape == kx.shape
    assert W.shape == (kx.size, kx.size)
    assert np.allclose(gamma, gamma_ref)
    # Identity modal matrix (no cross-order coupling in uniform layer)
    assert np.allclose(W, np.eye(kx.size))
