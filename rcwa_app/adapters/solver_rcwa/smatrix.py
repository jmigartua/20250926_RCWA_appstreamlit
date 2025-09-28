from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SMatrix:
    """
    2-port scattering matrix with blocks (all square, same size):
        [S11  S12]
        [S21  S22]
    Shapes: each (N, N), complex128.
    """

    S11: np.ndarray
    S12: np.ndarray
    S21: np.ndarray
    S22: np.ndarray

    @property
    def n(self) -> int:
        return int(self.S11.shape[0])


def _eye(n: int) -> np.ndarray:
    return np.eye(n, dtype=np.complex128)


def s_identity(n: int) -> SMatrix:
    """
    Identity 2-port (zero reflections, perfect transmission):
        S11 = S22 = 0,  S12 = S21 = I.
    It behaves as the neutral element for the star product.
    """
    Z = np.zeros((n, n), dtype=np.complex128)
    I_mat = _eye(n)
    return SMatrix(S11=Z, S12=I_mat, S21=I_mat, S22=Z)


def s_propagate(kz: np.ndarray, thickness_um: float) -> SMatrix:
    """
    Lossless segment propagation (no interface): phase advance only.
    Transmission in both directions is diag(exp(+i kz d)).
    Reflections are zero.

    kz: (N,) complex128 — modal z-wavenumbers (rad/μm).
    d : thickness in μm.
    """
    kz = np.asarray(kz, dtype=np.complex128)
    P = np.diag(np.exp(1j * kz * np.complex128(thickness_um)))
    Z = np.zeros_like(P)
    return SMatrix(S11=Z, S12=P, S21=P, S22=Z)


def redheffer_star(A: SMatrix, B: SMatrix, tau: float = 1e-12) -> SMatrix:
    """
    Redheffer star product: C = A ⋆ B
    Physically, B cascaded after A (A first, then B). Stable solve with ridge tau.

    C11 = A11 + A12 (I - B11 A22)^(-1) B11 A21
    C12 = A12 (I - B11 A22)^(-1) B12
    C21 = B21 (I - A22 B11)^(-1) A21
    C22 = B22 + B21 (I - A22 B11)^(-1) A22 B12
    """
    n = A.n
    I_mat = _eye(n)

    # Blocks for brevity
    A11, A12, A21, A22 = A.S11, A.S12, A.S21, A.S22
    B11, B12, B21, B22 = B.S11, B.S12, B.S21, B.S22

    # Stable solves
    M1 = I_mat - B11 @ A22 + tau * I_mat
    M2 = I_mat - A22 @ B11 + tau * I_mat
    X1 = np.linalg.solve(M1, I_mat)  # (I - B11 A22)^(-1)
    X2 = np.linalg.solve(M2, I_mat)  # (I - A22 B11)^(-1)

    C11 = A11 + A12 @ X1 @ B11 @ A21
    C12 = A12 @ X1 @ B12
    C21 = B21 @ X2 @ A21
    C22 = B22 + B21 @ X2 @ A22 @ B12

    return SMatrix(C11, C12, C21, C22)
