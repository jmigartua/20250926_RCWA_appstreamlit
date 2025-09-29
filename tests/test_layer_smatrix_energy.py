# tests/test_layer_smatrix_energy.py
from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.layer1d import slab_smatrix_order_basis


def _power_TE(
    r: np.ndarray, t: np.ndarray, kz_in: np.ndarray, kz_out: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # TE power factors: T = Re(kz_out)/Re(kz_in) * |t|^2, R = |r|^2 (for lossless cover)
    # Guard division by zero with tiny epsilon on purely evanescents.
    re_in = np.maximum(np.real(kz_in), 1e-18)
    R = np.abs(r) ** 2
    T = (np.real(kz_out) / re_in) * (np.abs(t) ** 2)
    return R, T


def test_slab_energy_conservation_normal_TE() -> None:
    # Single propagating order at normal incidence
    n_in = 1.0
    n_layer = 1.5
    n_out = 1.0
    lam = 5.0
    k0 = 2.0 * np.pi / lam
    kx = np.array([0.0 + 0.0j])  # m=0 only
    d = 0.8  # μm

    S = slab_smatrix_order_basis(
        pol="TE", n_in=n_in, n_layer=n_layer, n_out=n_out, k0=k0, kx=kx, thickness_um=d
    )

    r = np.diag(S.S11)
    t = np.diag(S.S12)
    kz_in = np.sqrt((n_in * k0) ** 2 - kx * kx + 0j)
    kz_out = np.sqrt((n_out * k0) ** 2 - kx * kx + 0j)

    R, T = _power_TE(r, t, kz_in, kz_out)
    assert np.allclose(R + T, 1.0, atol=1e-12)
