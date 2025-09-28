from __future__ import annotations

import numpy as np


def admittance_te(n: float | np.ndarray, kz: np.ndarray, k0: float) -> np.ndarray:
    """
    TE (s-polarization) wave admittance q = kz / (μ0 ω); in normalized units
    it is ∝ kz (we drop common constants since only ratios matter for power).
    Returns complex128, broadcast to kz.
    """
    kz = np.asarray(kz, dtype=np.complex128)
    _ = float(k0)  # kept for signature symmetry; constants cancel in ratios
    return kz.astype(np.complex128)


def admittance_tm(n: float | np.ndarray, kz: np.ndarray, k0: float) -> np.ndarray:
    """
    TM (p-polarization) normalized admittance q ∝ kz / n^2 (constants drop out).
    This follows the usual RCWA normalization used for power factors.
    """
    kz = np.asarray(kz, dtype=np.complex128)
    n2 = np.asarray(n, dtype=float) ** 2
    return kz / n2.astype(np.complex128)


def power_from_amplitudes(a: np.ndarray, q_mode: np.ndarray, q_incident: complex) -> np.ndarray:
    """
    Convert complex order amplitudes 'a' into *power fractions* using
    the admittance ratio Re(q_mode)/Re(q_incident). Clip small negatives.

        P_m = |a_m|^2 * Re(q_m) / Re(q_incident)

    Shapes:
      a: (M,) complex
      q_mode: (M,) complex
      q_incident: scalar complex
    """
    a = np.asarray(a, dtype=np.complex128)
    q_mode = np.asarray(q_mode, dtype=np.complex128)
    rq = np.real(q_mode)
    rq0 = (
        float(np.real(q_incident))
        if np.isrealobj(q_incident) is False
        else float(np.real(q_incident))
    )
    # avoid divide-by-zero but keep determinism
    if abs(rq0) < 1e-30:
        rq0 = 1e-30
    P = (np.abs(a) ** 2) * (rq / rq0)
    return np.clip(P.real, 0.0, np.inf)


def propagating_mask(k0: float, n: float, kx: np.ndarray) -> np.ndarray:
    """
    Boolean mask for propagating (radiative) orders in a homogeneous medium n:
      propagating if (n k0)^2 - |kx|^2 >= 0  (within small tolerance).
    """
    kx = np.asarray(kx, dtype=np.complex128)
    disc = (np.complex128(n) * np.complex128(k0)) ** 2 - (kx * kx)
    return (np.real(disc) >= -1e-12).astype(bool)
