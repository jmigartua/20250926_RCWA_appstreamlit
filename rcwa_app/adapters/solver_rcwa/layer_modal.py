# rcwa_app/adapters/solver_rcwa/layer_modal.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# ----------------------------
# Utilities and S-matrix types
# ----------------------------


@dataclass
class SMatrix:
    S11: np.ndarray
    S12: np.ndarray
    S21: np.ndarray
    S22: np.ndarray


def _new_SMatrix(S11: np.ndarray, S12: np.ndarray, S21: np.ndarray, S22: np.ndarray) -> SMatrix:
    """Ensure consistent dtype/shape for S-matrix blocks."""
    return SMatrix(
        S11=np.asarray(S11, dtype=np.complex128),
        S12=np.asarray(S12, dtype=np.complex128),
        S21=np.asarray(S21, dtype=np.complex128),
        S22=np.asarray(S22, dtype=np.complex128),
    )


def _principal_kz(n: float, k0: float, kx: np.ndarray) -> np.ndarray:
    """
    Principal-branch longitudinal wavenumber:
      kz = sqrt((n k0)^2 - kx^2) with Im(kz) >= 0 for evanescent orders.
    """
    n = float(n)
    k0 = float(k0)
    kx = np.asarray(kx, dtype=np.complex128)
    kz2 = (n * k0) ** 2 - kx * kx
    # Principal branch: real kz for propagating; +j|.| for evanescent
    kz = np.where(np.real(kz2) >= 0, np.sqrt(kz2 + 0j), 1j * np.sqrt(-kz2 + 0j))
    return kz.astype(np.complex128)


def _s_propagate(gamma: np.ndarray, thickness_um: float) -> SMatrix:
    """
    Propagation S-matrix for a uniform segment of thickness d with longitudinal
    modal constants 'gamma' (here gamma ≡ kz). No reflections; symmetric transmission.
    """
    phase = np.exp(1j * np.asarray(gamma, dtype=np.complex128) * float(thickness_um))
    P = np.diag(phase)
    Z = np.zeros_like(P)
    return _new_SMatrix(S11=Z, S12=P, S21=P, S22=Z)


def s_starprod(A: SMatrix, B: SMatrix) -> SMatrix:
    """
    Redheffer star product: C = A ⊛ B  (A first, then B).
    Stable implementation via linear solves (no explicit inverses).
    """
    n = A.S12.shape[1]
    I_mat = np.eye(n, dtype=np.complex128)

    M1 = I_mat - B.S11 @ A.S22
    C11 = A.S11 + A.S12 @ np.linalg.solve(M1, B.S11 @ A.S21)
    C12 = A.S12 @ np.linalg.solve(M1, B.S12)

    M2 = I_mat - A.S22 @ B.S11
    C21 = B.S21 @ np.linalg.solve(M2, A.S21)
    C22 = B.S22 + B.S21 @ np.linalg.solve(M2, A.S22 @ B.S12)

    return _new_SMatrix(C11, C12, C21, C22)


# ----------------------------
# Admittances and interfaces
# ----------------------------


def _admittance_order_TE(n: float, k0: float, kx: np.ndarray) -> np.ndarray:
    """Order-basis TE admittance: Y = kz / k0."""
    kz = _principal_kz(n, k0, kx)
    return np.diag(kz / float(k0))


def _admittance_order_TM(n: float, k0: float, kx: np.ndarray) -> np.ndarray:
    """
    Order-basis TM admittance (assuming μ=1): Y = kz / (k0 * ε) = kz / (k0 * n^2).
    This matches the standard RCWA TM continuity choice.
    """
    kz = _principal_kz(n, k0, kx)
    return np.diag(kz / (float(k0) * (float(n) ** 2)))


def _interface_from_Y(YL: np.ndarray, YR: np.ndarray) -> SMatrix:
    """
    Interface S from left/right admittances (same basis on both sides):

      S11 = (YL - YR)(YL + YR)^{-1}
      S12 = 2 YL (YL + YR)^{-1}
      S21 = 2 YR (YL + YR)^{-1}
      S22 = (YR - YL)(YL + YR)^{-1}
    """
    YL = np.asarray(YL, dtype=np.complex128)
    YR = np.asarray(YR, dtype=np.complex128)
    n = YL.shape[0]
    I_mat = np.eye(n, dtype=np.complex128)

    M = YL + YR  # solve on the right
    MinvI = np.linalg.solve(M, I_mat)

    S11 = (YL - YR) @ MinvI
    S12 = (2.0 * YL) @ MinvI
    S21 = (2.0 * YR) @ MinvI
    S22 = (YR - YL) @ MinvI
    return _new_SMatrix(S11, S12, S21, S22)


# ----------------------------
# Local slab (order-basis) reference — no imports needed
# ----------------------------


def _slab_smatrix_order_local(
    *,
    pol: Literal["TE", "TM"],
    n_in: float,
    n_layer: float,
    n_out: float,
    k0: float,
    kx: np.ndarray,
    thickness_um: float,
) -> SMatrix:
    """
    Build the uniform slab S-matrix in the order basis (TE/TM) using interface admittances.
    This mirrors typical textbook RCWA slab construction.
    """
    k0 = float(k0)
    kx = np.asarray(kx, dtype=np.complex128)
    # N = kx.size

    # Admittances on both sides and in the layer
    if pol == "TE":
        Y_in = _admittance_order_TE(n_in, k0, kx)
        Y_la = _admittance_order_TE(n_layer, k0, kx)
        Y_out = _admittance_order_TE(n_out, k0, kx)
    else:  # TM
        Y_in = _admittance_order_TM(n_in, k0, kx)
        Y_la = _admittance_order_TM(n_layer, k0, kx)
        Y_out = _admittance_order_TM(n_out, k0, kx)

    # Interfaces and propagation
    S_L = _interface_from_Y(Y_in, Y_la)
    kz_la = _principal_kz(n_layer, k0, kx)
    S_P = _s_propagate(kz_la, float(thickness_um))
    S_R = _interface_from_Y(Y_la, Y_out)

    return s_starprod(s_starprod(S_L, S_P), S_R)


# ----------------------------
# Public APIs
# ----------------------------


def layer_modal_te_smatrix(
    *,
    W: np.ndarray,
    gamma: np.ndarray,
    thickness_um: float,
    k0: float,
    kx: np.ndarray,
    n_in: float,
    n_out: float,
    boundary_model: Literal["li-te", "uniform-fallback"],
    n_layer_uniform: float | None = None,
) -> SMatrix:
    """
    TE layer S matrix in the *order* basis via a modal representation inside the layer.

    boundary_model:
      - "li-te":    Li TE continuity with coupling via W.
      - "uniform-fallback": compute the exact order-basis slab S (no external import).
    """
    k0 = float(k0)
    kx = np.asarray(kx, dtype=np.complex128)
    W = np.asarray(W, dtype=np.complex128)
    gamma = np.asarray(gamma, dtype=np.complex128)
    N = kx.size

    if boundary_model == "uniform-fallback":
        if n_layer_uniform is None:
            raise ValueError("n_layer_uniform is required for boundary_model='uniform-fallback'")
        return _slab_smatrix_order_local(
            pol="TE",
            n_in=float(n_in),
            n_layer=float(n_layer_uniform),
            n_out=float(n_out),
            k0=k0,
            kx=kx,
            thickness_um=float(thickness_um),
        )

    # Li TE boundary model
    Yo_in = _admittance_order_TE(float(n_in), k0, kx)
    Yo_out = _admittance_order_TE(float(n_out), k0, kx)

    # *** Key fix: modal admittance must be gamma/k0 to match order-basis scaling ***
    Ym = np.diag(gamma / k0)

    I_mat = np.eye(N, dtype=np.complex128)
    if np.allclose(W, I_mat, atol=1e-12, rtol=0.0):
        # Exactly reduce to slab (two interfaces + propagation)
        S_L = _interface_from_Y(Yo_in, Ym)
        S_P = _s_propagate(gamma, float(thickness_um))
        S_R = _interface_from_Y(Ym, Yo_out)
        return s_starprod(s_starprod(S_L, S_P), S_R)

    # General coupling via W (map modal admittance into order basis)
    W_inv = np.linalg.inv(W)
    Y_modal_in_order = (W @ Ym) @ W_inv

    S_L = _interface_from_Y(Yo_in, Y_modal_in_order)
    S_P = _s_propagate(gamma, float(thickness_um))
    S_R = _interface_from_Y(Y_modal_in_order, Yo_out)

    return s_starprod(s_starprod(S_L, S_P), S_R)


def layer_modal_tm_smatrix(
    *,
    W: np.ndarray,
    gamma: np.ndarray,
    thickness_um: float,
    k0: float,
    kx: np.ndarray,
    n_in: float,
    n_out: float,
    boundary_model: Literal["uniform-fallback"],
    n_layer_uniform: float | None = None,
) -> SMatrix:
    """
    TM variant.
    For now we expose only the 'uniform-fallback' (exact slab). When ready, add Lalanne TM
    boundary maps mirroring the Li TE structure (with η-harmonics).
    """
    k0 = float(k0)
    kx = np.asarray(kx, dtype=np.complex128)

    if boundary_model != "uniform-fallback":
        raise NotImplementedError("TM Li/Lalanne interface not wired yet; use 'uniform-fallback'.")

    if n_layer_uniform is None:
        raise ValueError("n_layer_uniform is required for boundary_model='uniform-fallback'")

    return _slab_smatrix_order_local(
        pol="TM",
        n_in=float(n_in),
        n_layer=float(n_layer_uniform),
        n_out=float(n_out),
        k0=k0,
        kx=kx,
        thickness_um=float(thickness_um),
    )


def layer_modal_te_from_profile(
    *,
    eps_hi: float,
    eps_lo: float,
    duty: float,
    M: int,
    lambda_um: float,
    theta_deg: float,
    period_um: float,
    n_in: float,
    n_out: float,
    thickness_um: float,
) -> SMatrix:
    """
    One-call TE layer S-matrix from a lamellar profile using Li TE boundary model.

    Notes:
      • Uses eigs_te_from_profile(...) to get (W, gamma, kx).
      • The order-grid β = n_in * k0 * sin(theta) (so 'n_in' is used as the ambient for kx).
      • Interfaces are Li TE (already implemented in layer_modal_te_smatrix).
    """
    # Local import to avoid top-level dependency ripple
    from .rigorous1d import eigs_te_from_profile  # assumes this module is present and typed

    gamma, W, kx = eigs_te_from_profile(
        eps_hi=eps_hi,
        eps_lo=eps_lo,
        duty=duty,
        M=M,
        lambda_um=lambda_um,
        theta_deg=theta_deg,
        period_um=period_um,
        n_ambient=n_in,  # keep the kx-consistency with the input side
    )

    k0 = 2.0 * np.pi / float(lambda_um)

    return layer_modal_te_smatrix(
        W=W,
        gamma=gamma,
        thickness_um=float(thickness_um),
        k0=float(k0),
        kx=kx,
        n_in=float(n_in),
        n_out=float(n_out),
        boundary_model="li-te",
    )


def stack_starprod(layers: list[SMatrix]) -> SMatrix:
    """
    Compose a sequence of S-matrices: S_total = layers[0] ⊛ layers[1] ⊛ ... ⊛ layers[-1]
    Safe on empty/one-length inputs.
    """
    if not layers:
        raise ValueError("stack_starprod: need at least one layer S-matrix")

    S_total = layers[0]
    for S in layers[1:]:
        S_total = s_starprod(S_total, S)
    return S_total
