from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.smatrix import SMatrix, redheffer_star, s_identity, s_propagate


def _rand_unit_phases(n: int) -> np.ndarray:
    rng = np.random.default_rng(123)
    phi = rng.uniform(-np.pi, np.pi, size=n)
    return np.diag(np.exp(1j * phi).astype(np.complex128))


def test_identity_is_neutral_element() -> None:
    n = 4
    # Make a random but well-conditioned 2-port with no reflections (for simplicity)
    T = _rand_unit_phases(n)
    A = SMatrix(
        S11=np.zeros((n, n), dtype=np.complex128),
        S12=T,
        S21=T,
        S22=np.zeros((n, n), dtype=np.complex128),
    )
    S_id = s_identity(n)

    left = redheffer_star(S_id, A)
    right = redheffer_star(A, S_id)

    assert np.allclose(left.S11, A.S11)
    assert np.allclose(left.S12, A.S12)
    assert np.allclose(left.S21, A.S21)
    assert np.allclose(left.S22, A.S22)

    assert np.allclose(right.S11, A.S11)
    assert np.allclose(right.S12, A.S12)
    assert np.allclose(right.S21, A.S21)
    assert np.allclose(right.S22, A.S22)


def test_lossless_propagator_is_unit_modulus() -> None:
    n = 5
    kz = np.linspace(1.0, 2.0, n).astype(np.complex128)  # rad/μm
    d = 3.5  # μm
    P = s_propagate(kz, d)
    # no reflections
    assert np.allclose(P.S11, 0.0)
    assert np.allclose(P.S22, 0.0)
    # transmission magnitude 1 (unitary diagonal)
    mag12 = np.abs(np.diag(P.S12))
    mag21 = np.abs(np.diag(P.S21))
    assert np.allclose(mag12, 1.0)
    assert np.allclose(mag21, 1.0)
