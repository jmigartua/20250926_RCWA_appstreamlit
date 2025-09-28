from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import (
    LamellarFourier,
    li_factor_operator_te,
    li_factor_operator_tm,
)


def test_uniform_limit_inverse_is_identity_scaled() -> None:
    M = 4
    eps0 = 2.25  # n=1.5
    # Uniform profile => eps_hi == eps_lo == eps0, any duty
    lf = LamellarFourier(eps_hi=eps0, eps_lo=eps0)
    h = lf.eps_harmonics(duty=0.37, M=M)  # duty arbitrary in uniform case
    Cinv_te = li_factor_operator_te(h[M:], tau=1e-14)
    Cinv_tm = li_factor_operator_tm(h[M:], tau=1e-14)

    I_mat = np.eye(2 * M + 1, dtype=float)
    assert np.allclose(Cinv_te, (1.0 / eps0) * I_mat, rtol=1e-12, atol=1e-12)
    assert np.allclose(Cinv_tm, (1.0 / eps0) * I_mat, rtol=1e-12, atol=1e-12)
