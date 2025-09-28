from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.smatrix import redheffer_star, s_propagate


def test_star_associativity_on_lossless_segments() -> None:
    # For pure propagation segments (no reflections), ⋆ is associative.
    kz = np.array([1.0, 1.2, 1.5], dtype=np.complex128)
    A = s_propagate(kz, 0.7)
    B = s_propagate(kz, 1.1)
    C = s_propagate(kz, 0.4)

    left = redheffer_star(A, redheffer_star(B, C))
    right = redheffer_star(redheffer_star(A, B), C)

    assert np.allclose(left.S12, right.S12)
    assert np.allclose(left.S21, right.S21)
    assert np.allclose(left.S11, right.S11)
    assert np.allclose(left.S22, right.S22)
