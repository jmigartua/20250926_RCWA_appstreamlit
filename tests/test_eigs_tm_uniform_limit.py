from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import (
    eigs_tm_from_profile,
    kz_from_dispersion,
)


def test_eigs_tm_matches_dispersion_in_uniform_limit() -> None:
    # Uniform slab: TM == TE in dispersion
    eps0 = 2.25  # n = 1.5
    lam = 5.0
    theta = 13.0
    period = 10.0
    M = 4
    n_ambient = 1.0

    gamma, W, kx = eigs_tm_from_profile(
        eps_hi=eps0,
        eps_lo=eps0,
        duty=0.5,
        M=M,
        lambda_um=lam,
        theta_deg=theta,
        period_um=period,
        n_ambient=n_ambient,
    )

    # Reference planar kz for each order
    k0 = 2.0 * np.pi / lam
    n = float(np.sqrt(eps0))
    kz_ref = kz_from_dispersion(k0, n, kx)

    # Compare after sorting by Re(kz), as in TE test
    idx = np.argsort(np.real(kz_ref))
    assert np.allclose(gamma[idx], kz_ref[idx], rtol=1e-10, atol=1e-12)

    # Orthonormal columns expected from Hermitian operator path in the uniform limit
    gram = W.conj().T @ W
    assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-12)
