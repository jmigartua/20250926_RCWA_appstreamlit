# tests/test_tm_operator_contract.py
import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import LamellarFourier, operator_tm_from_harmonics


def test_tm_operator_contract_shapes() -> None:
    M = 3
    lf = LamellarFourier(eps_hi=4.0, eps_lo=2.0)
    h_eta_full = lf.eta_harmonics(duty=0.4, M=M)
    h_eta = h_eta_full[M:]  # [η0..ηM]
    kx = np.linspace(-1.0, 1.0, 2 * M + 1).astype(np.complex128)
    A = operator_tm_from_harmonics(h_eta_nonneg=h_eta, kx=kx, k0=1.0)
    assert A.shape == (2 * M + 1, 2 * M + 1)
    # Hermitian check
    assert np.allclose(A, A.conj().T, atol=1e-12)
