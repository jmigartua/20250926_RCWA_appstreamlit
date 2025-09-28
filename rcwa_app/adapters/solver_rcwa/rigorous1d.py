from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

Pol = Literal["TE", "TM", "UNPOL"]


# ----------------------------- Fourier builder ---------------------------------------
class FourierBuilder(Protocol):
    def eps_fourier(self, *, duty: float, M: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (diag_eps_g, conv_eps) for a 1D lamellar profile with given duty.

        diag_eps_g: (2M+1,) vector with the *zeroth* coefficient duplicated across
                    the diagonal convenience vector used by some RCWA formulations.
        conv_eps:   (2M+1, 2M+1) Toeplitz convolution matrix C such that
                    (C @ f)_m = \\sum_g eps_g f_{m-g}.
        """


def toeplitz_from_harmonics(h: np.ndarray) -> np.ndarray:
    """
    Build a real symmetric Toeplitz convolution matrix from the non-negative
    harmonics h = [h0, h1, ..., h_M]. Returns C with shape (2M+1, 2M+1) and
    C[i, j] = h[|i - j|] when |i - j| ≤ M, else 0.

    This matches the standard RCWA convention when we pass the slice h[M:].
    """
    if h.ndim != 1 or h.size < 1:
        raise ValueError("h must be a 1D array with at least one element (h0).")
    M = h.size - 1
    size = 2 * M + 1
    C = np.empty((size, size), dtype=float)
    for i in range(size):
        for j in range(size):
            k = abs(i - j)
            C[i, j] = h[k] if k <= M else 0.0
    return C


@dataclass(frozen=True)
class LamellarFourier:
    """Analytical Fourier series for a binary (lamellar) permittivity profile.

    eps(x) = eps_hi on a fraction `duty` of the period, and eps_lo otherwise.
    The complex permittivity is taken real here; absorption enters later in the
    rigorous engine through material indices.
    """

    eps_hi: float
    eps_lo: float

    def eps_harmonics(self, *, duty: float, M: int) -> np.ndarray:
        """Return harmonics eps_g for g=-M..+M (length 2M+1).

        eps_0 = eps_lo + (eps_hi - eps_lo) * duty
        eps_g (g != 0) = (eps_hi - eps_lo) * sin(pi * g * duty) / (pi * g)
        """
        if not (0.0 <= duty <= 1.0):
            raise ValueError("duty must be in [0, 1]")
        g = np.arange(-M, M + 1, dtype=int)
        h = np.empty_like(g, dtype=float)
        delta = float(self.eps_hi - self.eps_lo)
        # Zeroth coefficient
        eps0 = float(self.eps_lo + delta * duty)
        h[g == 0] = eps0
        # Off-diagonals (sinc-style)
        g_nz = g[g != 0].astype(float)
        h[g != 0] = delta * np.sin(np.pi * g_nz * duty) / (np.pi * g_nz)
        return h

    def eps_fourier(self, *, duty: float, M: int) -> tuple[np.ndarray, np.ndarray]:
        h = self.eps_harmonics(duty=duty, M=M)
        diag = np.full_like(h, fill_value=h[(h.size - 1) // 2], dtype=float)  # eps0 vector
        C = toeplitz_from_harmonics(h[(h.size - 1) // 2 :])  # pass non-negative g (|i-j|) slice
        # The slice h[M:] creates an array [h0, h1, h2, ...] so C[i,j] = h[|i-j|]
        return diag, C


# ----------------------------- Order grid & kx utilities -----------------------------


def symmetric_orders(N: int) -> np.ndarray:
    """Return symmetric order set with an explicit m=0.

    Guarantees an odd count and inclusion of m=0. For even N it will widen the
    set to keep symmetry and include zero.
    """
    M = max(1, N // 2)
    m = np.arange(-M, M + 1, dtype=int)
    if (m == 0).sum() == 0:
        m = np.arange(-N, N + 1, dtype=int)
    return m


def kx_orders(
    lambda_um: float, theta_deg: float, period_um: float, m: np.ndarray, n_medium: float = 1.0
) -> np.ndarray:
    """In-plane wavevector components (x-direction) for order set m.

    kx_m = k0 * (n_medium * sin(theta) + m * lambda / period)
    where k0 = 2π/λ. The result is in rad/μm.
    """
    k0 = 2.0 * np.pi / float(lambda_um)
    s = float(n_medium) * np.sin(np.deg2rad(float(theta_deg)))
    return k0 * (s + (float(lambda_um) / float(period_um)) * m.astype(float))


# ----------------------------- Modal utilities (uniform layer) ------------------------


def _safe_complex_sqrt(z: np.ndarray | complex) -> np.ndarray:
    """
    Principal-branch complex square root, vectorized.
    Ensures dtype=complex128 and avoids warnings for tiny negatives via +0j.
    """
    return np.sqrt(np.asarray(z, dtype=np.complex128) + 0j)


def kz_from_dispersion(k0: float, n: float, kx: np.ndarray) -> np.ndarray:
    r"""
    Modal z-component for a *uniform* layer: k_z = sqrt( (k0 n)^2 - kx^2 ).
    Returns complex128 with principal branch.
    """
    kx = np.asarray(kx, dtype=np.complex128)
    k = np.complex128(k0) * np.complex128(n)
    return _safe_complex_sqrt(k * k - kx * kx)


def modal_uniform_te_tm(eps0: float, k0: float, kx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Uniform-layer modal solution for both TE/TM (no coupling between orders):
    - Inputs: avg permittivity eps0 (real, >=0), free-space k0, order grid kx.
    - Outputs:
        gamma: (2M+1,) complex array of k_z values
        W:     (2M+1, 2M+1) identity modal matrix (no mixing for uniform medium)
    """
    eps0 = float(eps0)
    n = float(np.sqrt(max(eps0, 0.0)))
    gamma = kz_from_dispersion(float(k0), n, np.asarray(kx))
    W = np.eye(gamma.size, dtype=np.complex128)
    return gamma, W


# ----------------------------- Factorized convolution operators ------------------------


def toeplitz_inverse_from_nonneg(h_nonneg: np.ndarray, tau: float = 1e-12) -> np.ndarray:
    """
    Stable inverse of the Toeplitz convolution matrix C built from non-negative
    harmonics h_nonneg = [h0, h1, ..., hM]. Returns C^{-1}.

    Regularizes with ridge tau to avoid blow-ups for near-singular cases.
    """
    if h_nonneg.ndim != 1 or h_nonneg.size < 1:
        raise ValueError("h_nonneg must be 1D with at least h0.")
    C = toeplitz_from_harmonics(h_nonneg)
    # Tikhonov: (C^T C + tau I)^{-1} C^T
    # Since C is symmetric, C^T C = C^2; we compute the ridge inverse directly.
    n = C.shape[0]
    reg = tau * np.eye(n, dtype=float)
    return np.linalg.solve(C + reg, np.eye(n, dtype=float))


def li_factor_operator_te(h_nonneg: np.ndarray, tau: float = 1e-12) -> np.ndarray:
    """
    Li's factorization for TE uses the convolution of eps^{-1}.
    Here we return C_eps^{-1} computed stably from h_nonneg.
    In the uniform limit (h1=h2=...=0), this equals (1/eps0) * I.
    """
    return toeplitz_inverse_from_nonneg(h_nonneg, tau=tau)


def li_factor_operator_tm(h_nonneg: np.ndarray, tau: float = 1e-12) -> np.ndarray:
    """
    Lalanne's rule for TM involves mixed products of eps and eps^{-1}.
    As a conservative scaffold (satisfying the uniform limit and symmetry),
    we return C_eps^{-1} as well; the full mixed form will replace this in Patch 4.
    """
    return toeplitz_inverse_from_nonneg(h_nonneg, tau=tau)
