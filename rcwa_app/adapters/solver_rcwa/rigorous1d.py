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


def toeplitz_from_harmonics(h_nonneg: np.ndarray) -> np.ndarray:
    """
    Build a real symmetric Toeplitz convolution matrix C from the non-negative
    Fourier harmonics of a real, even profile: h_nonneg = [h0, h1, ..., hM].

    Contract:
      • h_nonneg: 1D, length L = M+1 ≥ 1 (float-like)
      • return:  (N, N) with N = 2*M+1, using C[i, j] = h_{|i-j|} and 0 outside range.
    """
    h = np.asarray(h_nonneg, dtype=float).ravel()
    if h.ndim != 1:
        raise ValueError("h_nonneg must be a 1D array")
    L = int(h.size)
    if L < 1:
        raise ValueError("h_nonneg length must be at least 1 (must include h0)")

    M = L - 1
    N = 2 * M + 1

    # Zero-padded lookup: indices 0..M map to h0..hM, indices >M map to 0
    lookup = np.zeros(2 * M + 1, dtype=float)
    lookup[: M + 1] = h

    # Toeplitz by absolute index difference
    idx = np.arange(N)
    d = np.abs(idx[:, None] - idx[None, :])  # values in [0, 2M]
    C = lookup[d]
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

    def eta_harmonics(self, duty: float, M: int) -> np.ndarray:
        """
        Full Fourier spectrum of η(x) = 1/ε(x) for a two-level 1D lamellar grating,
        ordered g = -M..+M (length 2M+1). For positive, real ε, coefficients are real
        and even. The DC term is the duty-weighted average of the two levels.
        """
        duty = float(np.clip(duty, 0.0, 1.0))
        M = int(M)
        eta_hi = 1.0 / float(self.eps_hi)
        eta_lo = 1.0 / float(self.eps_lo)

        out = np.zeros(2 * M + 1, dtype=float)

        # g = 0 (DC)
        out[M] = duty * eta_hi + (1.0 - duty) * eta_lo
        if M == 0:
            return out

        # g ≠ 0: real, even spectrum for two-level lamellar profile
        delta = eta_hi - eta_lo
        for n in range(1, M + 1):
            # 2 * (Δη * sin(π n duty) / (π n))  → cosine-series amplitude folded to ±n
            coeff = 2.0 * delta * (np.sin(np.pi * n * duty) / (np.pi * n))
            out[M + n] = coeff  # +n
            out[M - n] = coeff  # −n
        return out


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


# ----------------------------- TE modal eigenproblem (Li factorization) -----------------


def operator_te_from_harmonics(
    h_nonneg: np.ndarray,
    kx: np.ndarray,
    k0: float,
) -> np.ndarray:
    """
    A_TE = k0^2 * C_eps - Kx^2, with C_eps built from non-negative harmonics [h0…hM].
    Returns (N, N), N = 2*M+1 = len(kx).
    """
    C_eps = toeplitz_from_harmonics(h_nonneg)
    kx = np.asarray(kx, dtype=np.complex128)
    N = C_eps.shape[0]
    if kx.size != N:
        raise ValueError(
            f"operator_te_from_harmonics: len(kx)={kx.size} must equal 2*M+1={N} "
            "derived from the harmonics length."
        )
    k0c = np.complex128(k0)
    K2 = np.diag(kx * kx)
    A = (k0c * k0c) * C_eps.astype(np.complex128) - K2
    # Hermitian enforcement (numerical) – safe, cheap
    A = 0.5 * (A + A.conj().T)
    return A


def eigs_te_from_profile(
    eps_hi: float,
    eps_lo: float,
    duty: float,
    M: int,
    lambda_um: float,
    theta_deg: float,
    period_um: float,
    n_ambient: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the TE modal eigensystem for a 1D lamellar profile on the FULL order grid.

    Returns
    -------
    gamma : (2M+1,) complex128  modal kz  (principal-branch sqrt of eigenvalues)
    W     : (2M+1, 2M+1) complex128  eigenvectors (columns, orthonormal for Hermitian A)
    kx    : (2M+1,) complex128  in-plane order grid used to build the operator
    """
    # 1) Convolution matrix of ε from FULL harmonics (g = -M..+M)
    lf = LamellarFourier(eps_hi=float(eps_hi), eps_lo=float(eps_lo))
    h_full = lf.eps_harmonics(duty=float(duty), M=int(M))  # length 2M+1
    # Derive M from harmonics length to ensure consistency
    M_eff = int((h_full.size - 1) // 2)
    # 2) FULL order grid m = -M_eff..+M_eff (length 2M+1)
    m = np.arange(-M_eff, M_eff + 1, dtype=np.complex128)
    # 3) Incident in-plane wavevector and grating momentum
    k0 = 2.0 * np.pi / float(lambda_um)
    theta_rad = float(np.deg2rad(theta_deg))
    beta = np.complex128(n_ambient) * np.complex128(k0) * np.sin(theta_rad)  # n k0 sinθ
    G = 2.0 * np.pi / float(period_um)  # 2π/Λ
    # 4) FULL kx array: no propagating filter here; len(kx) == len(h_full)
    kx = beta + G * m

    # Build Hermitian TE operator and solve (pass non-negative harmonics [h0..hM])
    h_nonneg = h_full[M_eff:]  # [h0, h1, ..., hM]
    A = operator_te_from_harmonics(h_nonneg=h_nonneg, kx=kx, k0=k0)
    # Hermitian ⇒ eigh gives orthonormal eigenvectors and real eigenvalues (up to FP noise)
    evals, evecs = np.linalg.eigh(A)
    # Reorder eigenpairs to align with the diffraction-order basis (m = -M..M):
    # column j is assigned to the row index where |W_ij| is maximal.
    dom = np.argmax(np.abs(evecs), axis=0)  # shape (2M+1,)
    perm = np.argsort(dom)  # permutation to basis order
    evecs = evecs[:, perm]
    evals = evals[perm]

    gamma = _safe_complex_sqrt(evals.astype(np.complex128))
    return gamma, evecs.astype(np.complex128), kx


# ----------------------------- TM modal eigenproblem (scaffold) -----------------------------


def eigs_tm_from_profile(
    eps_hi: float,
    eps_lo: float,
    duty: float,
    M: int,
    lambda_um: float,
    theta_deg: float,
    period_um: float,
    n_ambient: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TM modal eigensystem on the FULL order grid (scaffold).

    Returns
    -------
    gamma : (2M+1,) complex128
        Modal kz (principal-branch square roots).
    W     : (2M+1, 2M+1) complex128
        Eigenvector matrix (columns).
    kx    : (2M+1,) complex128
        In-plane order grid.

    Notes
    -----
    - Uniform-limit exactness is enforced by falling back to the TE closed-form
      operator when the profile is uniform (eps_hi == eps_lo) or the duty
      effectively turns the grating into a uniform slab (duty in {0, 1}).
    - Non-uniform TM (true Lalanne mixed factorization) is introduced in R.3.
    """
    # Decide uniformity
    is_uniform = (abs(float(eps_hi) - float(eps_lo)) < 1e-15) or (duty <= 0.0) or (duty >= 1.0)
    if is_uniform:
        # TE and TM coincide in the uniform limit; reuse TE machinery
        return eigs_te_from_profile(
            eps_hi=eps_hi,
            eps_lo=eps_lo,
            duty=min(max(duty, 0.0), 1.0),
            M=M,
            lambda_um=lambda_um,
            theta_deg=theta_deg,
            period_um=period_um,
            n_ambient=n_ambient,
        )
    # Non-uniform TM (mixed factorization) lands in R.3
    raise NotImplementedError(
        "TM mixed factorization for non-uniform profiles will be added in R.3"
    )
