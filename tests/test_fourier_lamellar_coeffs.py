from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import LamellarFourier, toeplitz_from_harmonics


def test_lamellar_eps0_and_first_harmonics() -> None:
    M = 5
    duty = 0.3
    eps_hi, eps_lo = 4.0, 2.0
    lf = LamellarFourier(eps_hi=eps_hi, eps_lo=eps_lo)
    h = lf.eps_harmonics(duty=duty, M=M)

    # Zeroth coefficient equals area average
    eps0 = eps_lo + (eps_hi - eps_lo) * duty
    assert abs(h[M] - eps0) < 1e-12

    # First harmonic matches analytical sinc-like value
    expected_g1 = (eps_hi - eps_lo) * np.sin(np.pi * 1 * duty) / (np.pi * 1)
    assert abs(h[M + 1] - expected_g1) < 1e-12
    assert abs(h[M - 1] - expected_g1) < 1e-12  # symmetry


def test_convolution_toeplitz_properties() -> None:
    M = 3
    duty = 0.5
    lf = LamellarFourier(eps_hi=3.0, eps_lo=1.0)
    h = lf.eps_harmonics(duty=duty, M=M)

    # Build Toeplitz from non-negative harmonics [h0, h1, ...]
    C = toeplitz_from_harmonics(h[M:])
    assert C.shape == (2 * M + 1, 2 * M + 1)

    # Symmetry: C[i,j] == C[j,i] for real lamellar
    assert np.allclose(C, C.T)

    # Diagonal equals h0 everywhere
    assert np.allclose(np.diag(C), h[M])
