from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

Pol = Literal["TE", "TM", "UNPOL"]


# ----------------------------- Fourier builder ---------------------------------------
class FourierBuilder(Protocol):
    def eps_fourier(self, *, duty: float, M: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (diag_eps_g, conv_eps) for a 1D lamellar profile with given duty.

        diag_eps_g: (2M+1,) zeroth- and side-harmonic permittivity Fourier coeffs
        conv_eps: (2M+1, 2M+1) Toeplitz-like convolution matrix
        """


@dataclass(frozen=True)
class LamellarFourier:
    eps_hi: float
    eps_lo: float

    def eps_fourier(self, *, duty: float, M: int) -> tuple[np.ndarray, np.ndarray]:
        # Simple analytical Fourier series for a binary (lamellar) profile.
        # eps(x) = eps_hi on width 'duty', eps_lo otherwise over one period.
        m = np.arange(-M, M + 1)
        diag = np.empty_like(m, dtype=float)
        conv = np.zeros((2 * M + 1, 2 * M + 1), dtype=float)

        # Zeroth coefficient
        eps0 = self.eps_lo + (self.eps_hi - self.eps_lo) * duty
        diag[:] = eps0

        # Off-diagonal harmonics (sinc-like)
        for k in m:
            if k == 0:
                val = eps0
            else:
                val = (self.eps_hi - self.eps_lo) * np.sin(np.pi * k * duty) / (np.pi * k)
            idx = k + M
            conv[idx, M] = val
            conv[M, idx] = val
        # Toeplitz extend (symmetric for real lamellar)
        for i in range(2 * M + 1):
            for j in range(2 * M + 1):
                conv[i, j] = conv[abs(i - j), M] if i >= j else conv[abs(j - i), M]
        return diag, conv


# ----------------------------- S-matrix layer API -----------------------------------
class SMatrixLayer(Protocol):
    def propagate(self, field: np.ndarray) -> np.ndarray: ...


@dataclass
class IdentityLayer:
    """Placeholder layer used until the rigorous eigen-solver is plugged in."""

    size: int

    def propagate(self, field: np.ndarray) -> np.ndarray:
        return field


# ----------------------------- Utility: order grid ----------------------------------


def symmetric_orders(N: int) -> np.ndarray:
    """Return symmetric order set with an explicit m=0."""
    M = max(1, N // 2)
    m = np.arange(-M, M + 1, dtype=int)
    if (m == 0).sum() == 0:
        m = np.arange(-N, N + 1, dtype=int)
    return m
