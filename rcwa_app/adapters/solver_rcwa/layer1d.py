# rcwa_app/adapters/solver_rcwa/layer1d.py
from __future__ import annotations

from typing import Literal

import numpy as np

from .smatrix import SMatrix

Polarization = Literal["TE", "TM"]


def _principal_kz(n: complex | float, k0: float, kx: np.ndarray) -> np.ndarray:
    """
    kz = sqrt((n k0)^2 - kx^2), principal branch, with Im(kz) >= 0 for evanescents.
    Shapes:
      kx: (N,)
      return: (N,)
    """
    nck0 = np.complex128(n) * np.complex128(k0)
    val = nck0 * nck0 - kx * kx
    kz = np.sqrt(val + 0.0j)  # principal sqrt: Im >= 0 on negative real axis
    # For propagating waves, make Re(kz) >= 0 to keep a consistent "forward" direction
    sign = np.where(np.real(kz) >= 0.0, 1.0 + 0.0j, -1.0 + 0.0j)
    return kz * sign


def _rt_interface_te(
    kz1: np.ndarray,
    kz2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fresnel amplitudes at an interface for TE polarization.
      r = (kz1 - kz2) / (kz1 + kz2)
      t = 2 kz1 / (kz1 + kz2)
    All arrays shape: (N,)
    """
    denom = kz1 + kz2
    # tiny regularization to avoid division warnings for pathological cases
    denom = np.where(np.abs(denom) < 1e-18, denom + 1e-18, denom)
    r = (kz1 - kz2) / denom
    t = (2.0 * kz1) / denom
    return r, t


def _rt_interface_tm(
    kz1: np.ndarray,
    kz2: np.ndarray,
    eps1: complex | float,
    eps2: complex | float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fresnel amplitudes at an interface for TM polarization.
      r = (ε2 kz1 - ε1 kz2) / (ε2 kz1 + ε1 kz2)
      t = (2 ε2 kz1) / (ε2 kz1 + ε1 kz2)
    """
    eps1c = np.complex128(eps1)
    eps2c = np.complex128(eps2)
    num_r = eps2c * kz1 - eps1c * kz2
    den = eps2c * kz1 + eps1c * kz2
    den = np.where(np.abs(den) < 1e-18, den + 1e-18, den)
    r = num_r / den
    t = (2.0 * eps2c * kz1) / den
    return r, t


def _slab_rt_orderwise(
    pol: Polarization,
    n_in: complex | float,
    n_layer: complex | float,
    n_out: complex | float,
    k0: float,
    kx: np.ndarray,
    thickness_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Order-by-order complex amplitudes for a single slab:
      Left incidence → rL, tLR
      Right incidence → rR, tRL
    Shapes:
      kx: (N,)
      returns: rL, tLR, rR, tRL each with shape (N,)
    """
    eps_in = (np.complex128(n_in) ** 2).real  # μ=1, standard optics
    eps_layer = (np.complex128(n_layer) ** 2).real
    eps_out = (np.complex128(n_out) ** 2).real

    kz_in = _principal_kz(n_in, k0, kx)
    kz_l = _principal_kz(n_layer, k0, kx)
    kz_out = _principal_kz(n_out, k0, kx)

    if pol == "TE":
        r12, t12 = _rt_interface_te(kz_in, kz_l)
        r21, t21 = _rt_interface_te(kz_l, kz_in)
        r23, t23 = _rt_interface_te(kz_l, kz_out)
        r32, t32 = _rt_interface_te(kz_out, kz_l)
    else:  # TM
        r12, t12 = _rt_interface_tm(kz_in, kz_l, eps_in, eps_layer)
        r21, t21 = _rt_interface_tm(kz_l, kz_in, eps_layer, eps_in)
        r23, t23 = _rt_interface_tm(kz_l, kz_out, eps_layer, eps_out)
        r32, t32 = _rt_interface_tm(kz_out, kz_l, eps_out, eps_layer)

    phase = np.exp(1.0j * kz_l * np.complex128(thickness_um))
    phase2 = phase * phase

    den_left = 1.0 + r12 * r23 * phase2
    den_left = np.where(np.abs(den_left) < 1e-18, den_left + 1e-18, den_left)
    tLR = (t12 * t23 * phase) / den_left
    rL = (r12 + r23 * phase2) / den_left

    # Right incidence: swap (1↔3)
    den_right = 1.0 + r32 * r21 * phase2
    den_right = np.where(np.abs(den_right) < 1e-18, den_right + 1e-18, den_right)
    tRL = (t32 * t21 * phase) / den_right
    rR = (r32 + r21 * phase2) / den_right

    return rL, tLR, rR, tRL


def slab_smatrix_order_basis(
    *,
    pol: Polarization,
    n_in: complex | float,
    n_layer: complex | float,
    n_out: complex | float,
    k0: float,
    kx: np.ndarray,
    thickness_um: float,
) -> SMatrix:
    """
    Build an order-diagonal scattering matrix for a homogeneous slab.
    This lives entirely in the plane-wave order basis (no modal mixing).

    Returns an SMatrix with shape (N, N) blocks, where N = len(kx).
    """
    kx = np.asarray(kx, dtype=np.complex128).ravel()
    rL, tLR, rR, tRL = _slab_rt_orderwise(
        pol=pol,
        n_in=n_in,
        n_layer=n_layer,
        n_out=n_out,
        k0=float(k0),
        kx=kx,
        thickness_um=float(thickness_um),
    )
    # Diagonal in the order basis (no cross-coupling for a homogeneous slab)
    diag = lambda v: np.diag(v.astype(np.complex128))  # noqa: E731
    return SMatrix(S11=diag(rL), S12=diag(tLR), S21=diag(tRL), S22=diag(rR))
