from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from rcwa_app.adapters.solver_rcwa.rigorous1d import (
    LamellarFourier,
    operator_te_from_harmonics,
    toeplitz_from_harmonics,
)


def _is_toeplitz(C: NDArray[np.floating | np.complexfloating], atol: float = 1e-12) -> bool:
    C = np.asarray(C)
    n = C.shape[0]
    for d in range(-(n - 1), n):
        diag = np.diag(C, k=d)
        if not np.allclose(diag, diag[0], atol=atol, rtol=0.0):
            return False
    return True


def test_eps_and_eta_harmonics_even_and_real() -> None:
    M = 4
    lf = LamellarFourier(eps_hi=4.0, eps_lo=2.0)
    for duty in (0.3, 0.5, 0.7):
        h_eps = lf.eps_harmonics(duty=duty, M=M)
        h_eta = lf.eta_harmonics(duty=duty, M=M)
        # Real and even (full vector g=-M..+M)
        for h in (h_eps, h_eta):
            assert h.shape == (2 * M + 1,)
            assert np.allclose(h.imag if np.iscomplexobj(h) else 0.0, 0.0, atol=1e-14)
            assert np.allclose(h, h[::-1], atol=1e-14)


def test_toeplitz_from_nonneg_shape_and_symmetry() -> None:
    M = 3
    lf = LamellarFourier(eps_hi=3.0, eps_lo=1.0)
    h_full = lf.eps_harmonics(duty=0.5, M=M)
    h_nonneg = h_full[M:]  # [h0..hM]
    C = toeplitz_from_harmonics(h_nonneg)
    assert C.shape == (2 * M + 1, 2 * M + 1)
    assert np.allclose(C, C.T, atol=1e-14)
    assert _is_toeplitz(C, atol=1e-12)


def test_operator_te_size_mismatch_raises() -> None:
    M = 2
    lf = LamellarFourier(eps_hi=2.0, eps_lo=1.5)
    h_full = lf.eps_harmonics(duty=0.4, M=M)
    h_nonneg = h_full[M:]
    C = toeplitz_from_harmonics(h_nonneg)
    # Build a wrong-sized kx on purpose
    kx_bad = np.linspace(-1.0, 1.0, C.shape[0] - 1).astype(np.complex128)
    try:
        operator_te_from_harmonics(h_nonneg=h_nonneg, kx=kx_bad, k0=1.0)
    except ValueError as e:
        assert "len(kx)" in str(e)
    else:
        raise AssertionError("operator_te_from_harmonics must raise on size mismatch")
