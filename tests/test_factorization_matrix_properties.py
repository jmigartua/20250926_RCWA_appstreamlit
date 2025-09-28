from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import LamellarFourier, li_factor_operator_te


def test_te_operator_is_symmetric_positive_definite_for_positive_eps() -> None:
    M = 5
    lf = LamellarFourier(eps_hi=4.0, eps_lo=2.0)
    h = lf.eps_harmonics(duty=0.3, M=M)
    Cinv = li_factor_operator_te(h[M:], tau=1e-10)

    # Symmetry (floating tolerance)
    assert np.allclose(Cinv, Cinv.T, rtol=1e-12, atol=1e-12)

    # Positive definiteness (all eigenvalues > 0 with regularization)
    w = np.linalg.eigvalsh(0.5 * (Cinv + Cinv.T))
    assert np.all(w > 0.0)
