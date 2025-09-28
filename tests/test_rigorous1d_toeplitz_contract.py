from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import (
    LamellarFourier,
    operator_te_from_harmonics,
)


def test_toeplitz_contract_shapes() -> None:
    """
    Contract guard:
    - operator_te_from_harmonics() expects non-negative harmonics [h0..hM] (len M+1),
      and a full order grid kx of length (2M+1).
    - The resulting Toeplitz-based operator must be (2M+1) x (2M+1).
    """
    M = 3
    lf = LamellarFourier(eps_hi=4.0, eps_lo=2.0)
    h_full = lf.eps_harmonics(duty=0.5, M=M)  # length = 2M+1, ordered g=-M..+M
    h_nonneg = h_full[M:]  # [h0, h1, ..., hM]
    kx = np.linspace(-1.0, 1.0, 2 * M + 1).astype(np.complex128)

    A = operator_te_from_harmonics(h_nonneg=h_nonneg, kx=kx, k0=1.0)
    assert A.shape == (2 * M + 1, 2 * M + 1)
