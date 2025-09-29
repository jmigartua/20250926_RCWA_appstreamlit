# tests/test_layer_cascade_equals_single.py
from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.layer1d import slab_smatrix_order_basis
from rcwa_app.adapters.solver_rcwa.smatrix import redheffer_star


def test_two_half_slabs_equal_one_full_TE() -> None:
    n_in = 1.0
    n_layer = 1.7
    n_out = 1.0
    lam = 4.0
    k0 = 2.0 * np.pi / lam
    kx = np.array([0.0 + 0.0j])  # single order
    d_full = 0.6
    d_half = d_full / 2.0

    S_half = slab_smatrix_order_basis(
        pol="TE", n_in=n_in, n_layer=n_layer, n_out=n_out, k0=k0, kx=kx, thickness_um=d_half
    )
    S_cascaded = redheffer_star(S_half, S_half)

    S_full = slab_smatrix_order_basis(
        pol="TE", n_in=n_in, n_layer=n_layer, n_out=n_out, k0=k0, kx=kx, thickness_um=d_full
    )

    # Compare all blocks
    for A, B in [
        (S_cascaded.S11, S_full.S11),
        (S_cascaded.S12, S_full.S12),
        (S_cascaded.S21, S_full.S21),
        (S_cascaded.S22, S_full.S22),
    ]:
        assert np.allclose(A, B, atol=1e-12)
