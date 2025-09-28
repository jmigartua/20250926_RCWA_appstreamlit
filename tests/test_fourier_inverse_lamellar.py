from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import LamellarFourier, toeplitz_from_harmonics


def test_eta_harmonics_uniform_limit() -> None:
    M = 4
    eps0 = 2.25
    lf = LamellarFourier(eps_hi=eps0, eps_lo=eps0)
    h_eta = lf.eta_harmonics(duty=0.37, M=M)
    # In uniform profile, η is constant → only DC survives
    assert np.allclose(h_eta, np.pad([1.0 / eps0], (M, M)))


def test_eta_toeplitz_shape_and_symmetry() -> None:
    M = 3
    lf = LamellarFourier(eps_hi=4.0, eps_lo=2.0)
    h_eta_full = lf.eta_harmonics(duty=0.5, M=M)
    h_eta_nonneg = h_eta_full[M:]  # [η0..ηM]
    C_eta = toeplitz_from_harmonics(h_eta_nonneg)
    # Toeplitz size and symmetry/Hermiticity guard
    assert C_eta.shape == (2 * M + 1, 2 * M + 1)
    assert np.allclose(C_eta, C_eta.T, atol=1e-14)
