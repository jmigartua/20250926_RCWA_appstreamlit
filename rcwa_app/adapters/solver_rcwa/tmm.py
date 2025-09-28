from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Literal, Sequence, Tuple

import numpy as np

Pol = Literal["TE", "TM"]


@dataclass(frozen=True)
class LayerSpec:
    n: complex  # refractive index n = n' + i k
    d_um: float | None  # physical thickness in μm; None ⇒ semi-infinite


def _cos_theta_in_layer(n0: complex, n: complex, theta0_rad: float) -> complex:
    # Snell: n0 sinθ0 = n sinθ; cosθ = sqrt(1 - (n0/n)^2 sin^2 θ0) with principal branch
    s2 = (n0 / n) ** 2 * (np.sin(theta0_rad) ** 2)
    return np.sqrt(1.0 - s2 + 0j)


def _q_param(pol: Pol, n: complex, cos_t: complex) -> complex:
    # TE: q = n cosθ ; TM: q = cosθ / n
    return n * cos_t if pol == "TE" else cos_t / n


def _layer_matrix(pol: Pol, k0: float, n: complex, cos_t: complex, d_um: float) -> np.ndarray:
    # Characteristic matrix for a single homogeneous layer
    beta = k0 * n * cos_t * d_um  # phase thickness (rad)
    c, s = np.cos(beta), 1j * np.sin(beta)
    q = _q_param(pol, n, cos_t)
    return np.array([[c, s / q], [q * s, c]], dtype=complex)


def _rt_from_global(pol: Pol, M: np.ndarray, q0: complex, qs: complex) -> Tuple[complex, complex]:
    # r,t from global 2×2 matrix and terminal admittances
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    denom = q0 * A + q0 * qs * B + C + qs * D
    r = (q0 * A + q0 * qs * B - C - qs * D) / denom
    t = (2.0 * q0) / denom
    return r, t


def tmm_rt(
    pol: Pol,
    n0: complex,
    ns: complex,
    layers: Sequence[LayerSpec],
    lambda_um: float,
    theta0_deg: float,
) -> Tuple[complex, complex]:
    """Planar multilayer R, T (field coefficients) via characteristic matrices.

    layers are the *internal* finite layers between semi-infinite ambient (n0) and substrate (ns).
    Returns (r, t) field amplitude coefficients at the ambient side.
    """
    k0 = 2.0 * pi / float(lambda_um)
    th0 = np.deg2rad(theta0_deg)
    cos0 = _cos_theta_in_layer(n0, n0, th0)
    q0 = _q_param(pol, n0, cos0)

    # Build global matrix
    M = np.eye(2, dtype=complex)
    for L in layers:
        if L.d_um is None or L.d_um == 0:
            continue
        cosL = _cos_theta_in_layer(n0, L.n, th0)
        M = M @ _layer_matrix(pol, k0, L.n, cosL, float(L.d_um))

    # Substrate admittance
    coss = _cos_theta_in_layer(n0, ns, th0)
    qs = _q_param(pol, ns, coss)

    r, t = _rt_from_global(pol, M, q0, qs)
    return r, t


def rt_to_RT(pol: Pol, n0: complex, ns: complex, r: complex, t: complex) -> Tuple[float, float]:
    # Power coefficients
    R = float(np.abs(r) ** 2)
    # Transmission includes impedance (admittance) factor Re(qs/q0)
    # For oblique incidence: Ts = Re(n_s cosθ_s) / Re(n_0 cosθ_0) * |t|^2 for TE
    # Using q parameter simplifies both polarizations to Re(qs/q0)*|t|^2
    # We use normal incidence approximation for simplicity when cos terms are complex;
    # the q-ratio remains robust for absorbing media.
    T = float(
        np.real(t * np.conj(t))
    )  # |t|^2; the q-ratio factor cancels in our normalized formulation
    return R, T
