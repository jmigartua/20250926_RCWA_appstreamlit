from __future__ import annotations

import numpy as np
import pytest

from rcwa_app.adapters.solver_rcwa.rigorous1d import kx_orders, symmetric_orders


def test_symmetric_orders_contains_zero_and_is_odd() -> None:
    for N in (3, 4, 7):
        m = symmetric_orders(N)
        assert (m == 0).any()
        assert m.size % 2 == 1
        assert m[0] == -m[-1]


def test_kx_orders_scales_with_lambda_and_period() -> None:
    m = symmetric_orders(5)
    # Use m=0 and a nonzero theta so kx ∝ k0 ∝ 1/λ
    idx0 = int(np.where(m == 0)[0][0])
    kx1 = kx_orders(lambda_um=4.0, theta_deg=10.0, period_um=10.0, m=m)
    kx2 = kx_orders(lambda_um=5.0, theta_deg=10.0, period_um=10.0, m=m)
    assert kx2[idx0] / kx1[idx0] == pytest.approx(4.0 / 5.0, rel=1e-12)
