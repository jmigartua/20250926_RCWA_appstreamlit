from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import eigs_te_from_profile


def test_eigs_te_eigenvectors_well_conditioned() -> None:
    # Non-uniform lamellar profile, representative parameters
    gamma, W, _ = eigs_te_from_profile(
        eps_hi=4.0,
        eps_lo=2.0,
        duty=0.5,
        M=5,
        lambda_um=6.0,
        theta_deg=15.0,
        period_um=8.0,
        n_ambient=1.0,
    )
    # eigh returns orthonormal columns; condition number ~ 1
    s = np.linalg.svd(W, compute_uv=False)
    cond = float(s.max() / max(1e-300, s.min()))
    assert cond < 1e2  # very conservative; typical ≈ 1
