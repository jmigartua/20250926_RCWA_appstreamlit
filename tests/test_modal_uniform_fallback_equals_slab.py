# tests/test_modal_uniform_fallback_equals_slab.py
from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.layer1d import _principal_kz, slab_smatrix_order_basis
from rcwa_app.adapters.solver_rcwa.layer_modal import layer_modal_te_smatrix, layer_modal_tm_smatrix


def _W_identity(N: int) -> np.ndarray:
    return np.eye(N, dtype=np.complex128)


def _gamma_uniform(n_layer: float, k0: float, kx: np.ndarray) -> np.ndarray:
    return _principal_kz(n_layer, k0, kx)


def test_modal_uniform_fallback_matches_slab_TE_TM() -> None:
    n_in, n_layer, n_out = 1.0, 1.5, 1.0
    lam = 4.2
    k0 = 2.0 * np.pi / lam
    # a few orders just to exercise shapes
    kx = np.array([-0.1 + 0.0j, 0.0 + 0.0j, 0.22 + 0.0j])
    d = 0.35

    W = _W_identity(len(kx))
    gamma = _gamma_uniform(n_layer, k0, kx)

    # Reference slab
    S_te_ref = slab_smatrix_order_basis(
        pol="TE", n_in=n_in, n_layer=n_layer, n_out=n_out, k0=k0, kx=kx, thickness_um=d
    )
    S_tm_ref = slab_smatrix_order_basis(
        pol="TM", n_in=n_in, n_layer=n_layer, n_out=n_out, k0=k0, kx=kx, thickness_um=d
    )

    # Modal fallback
    S_te = layer_modal_te_smatrix(
        W=W,
        gamma=gamma,
        thickness_um=d,
        k0=k0,
        kx=kx,
        n_in=n_in,
        n_out=n_out,
        boundary_model="uniform-fallback",
        n_layer_uniform=n_layer,
    )
    S_tm = layer_modal_tm_smatrix(
        W=W,
        gamma=gamma,
        thickness_um=d,
        k0=k0,
        kx=kx,
        n_in=n_in,
        n_out=n_out,
        boundary_model="uniform-fallback",
        n_layer_uniform=n_layer,
    )

    for A, B in [
        (S_te.S11, S_te_ref.S11),
        (S_te.S12, S_te_ref.S12),
        (S_te.S21, S_te_ref.S21),
        (S_te.S22, S_te_ref.S22),
        (S_tm.S11, S_tm_ref.S11),
        (S_tm.S12, S_tm_ref.S12),
        (S_tm.S21, S_tm_ref.S21),
        (S_tm.S22, S_tm_ref.S22),
    ]:
        assert np.allclose(A, B, atol=1e-12)
