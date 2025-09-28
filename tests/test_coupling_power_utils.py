from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.coupling import (
    admittance_te,
    admittance_tm,
    power_from_amplitudes,
    propagating_mask,
)
from rcwa_app.adapters.solver_rcwa.rigorous1d import kx_orders, symmetric_orders


def test_propagating_mask_matches_grating_equation() -> None:
    # Λ = 10 μm, λ = 5 μm, θ = 0°, n=1 ⇒ allowed |m| ≤ 2 since |m| λ/Λ ≤ 1
    lam, per, theta, n = 5.0, 10.0, 0.0, 1.0
    m = symmetric_orders(9)  # ...,-4,-3,-2,-1,0,1,2,3,4
    kx = kx_orders(lambda_um=lam, theta_deg=theta, period_um=per, m=m, n_medium=n)
    k0 = 2.0 * np.pi / lam
    mask = propagating_mask(k0, n, kx)
    # Only |m| ≤ 2 should propagate at θ=0°
    allowed = np.abs(m) <= int(per / lam)
    assert np.array_equal(mask, allowed)


def test_power_from_amplitudes_identity_case() -> None:
    # Single non-zero order amplitude, equal admittances => power is |a|^2
    a = np.zeros(5, dtype=np.complex128)
    a[2] = 0.3 + 0.4j  # |a|^2 = 0.25
    q_mode = np.ones_like(a, dtype=np.complex128)
    q0 = 1.0 + 0.0j
    P = power_from_amplitudes(a, q_mode, q0)
    assert P.shape == a.shape
    assert abs(P.sum() - 0.25) < 1e-15


def test_admittance_tm_te_simple_consistency() -> None:
    # For n=1, TE and TM reduce to the same up to the 1/n^2 factor
    kz = np.array([1.0, 2.0, 3.0], dtype=np.complex128)
    k0 = 2.0 * np.pi
    n = 1.0
    q_te = admittance_te(n, kz, k0)
    q_tm = admittance_tm(n, kz, k0)
    assert np.allclose(q_tm, q_te)  # since n=1 ⇒ n^2 = 1
