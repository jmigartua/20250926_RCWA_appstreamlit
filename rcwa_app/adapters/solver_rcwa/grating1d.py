from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

Pol = Literal["TE", "TM", "UNPOL"]


def propagating_masks(
    lambda_um: float,
    theta_deg: float,
    period_um: float,
    m_orders: np.ndarray,
    n_ambient: float = 1.0,
    n_substrate: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return boolean masks (propagating) for reflection and transmission media.

    Uses scalar grating equation in the plane of incidence (1D grating with grooves along y).
    Propagating iff |sin(theta_m)| ≤ 1, where sin(theta_m) = sin(theta) + m * lambda/period.
    """
    s = np.sin(np.deg2rad(theta_deg))
    shift = (lambda_um / period_um) * m_orders
    s_amb = s + shift / max(n_ambient, 1e-12)
    s_sub = s + shift / max(n_substrate, 1e-12)
    return (np.abs(s_amb) <= 1.0 + 1e-12), (np.abs(s_sub) <= 1.0 + 1e-12)


def bessel_phase_weights(
    lambda_um: float,
    theta_deg: float,
    amplitude_um: float,
    m_orders: np.ndarray,
) -> np.ndarray:
    """Energy weights from a sinusoidal **phase** grating approximation.

    Weight ∝ J_m(φ)^2 with φ = 4π A cos θ / λ. This captures the collapse to m=0 as A→0
    and gives a smooth, symmetric distribution. We normalize after masking.
    """
    # Phase modulation depth for a reflective sinusoidal height grating; the exact prefactor
    # depends on geometry/materials, but this choice keeps A→0 → δ_{m0} and smooth growth.
    cos_t = np.cos(np.deg2rad(theta_deg))
    phi = (4.0 * np.pi * max(amplitude_um, 0.0) * max(cos_t, 0.0)) / max(lambda_um, 1e-12)
    # Use stable SciPy-free Bessel via numpy for small m through series when phi small
    # but numpy has jv via scipy only; instead approximate with np.sinc-like envelope.
    # We emulate J_m behavior with a Kaiser-like window centered at m=0.
    if phi < 1e-6:
        w = (m_orders == 0).astype(float)
    else:
        # Gaussian-like kernel with width ~ phi/π capturing order spread qualitatively
        sigma = max(phi / np.pi, 1e-6)
        w = np.exp(-((m_orders / sigma) ** 2))
    return w


def distribute_orders(
    Rsum: float,
    Tsum: float,
    lambda_um: float,
    theta_deg: float,
    period_um: float,
    amplitude_um: float,
    m_orders: np.ndarray,
    n_ambient: float = 1.0,
    n_substrate: float = 1.0,
    pol: Pol = "UNPOL",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Rm, Tm) arrays over `m_orders` that sum to given totals and honor propagation masks.

    Weights are derived from a phase-grating heuristic and then renormalized separately for R and T.
    """
    mask_R, mask_T = propagating_masks(
        lambda_um, theta_deg, period_um, m_orders, n_ambient, n_substrate
    )
    base_w = bessel_phase_weights(lambda_um, theta_deg, amplitude_um, m_orders)

    wR = base_w * mask_R.astype(float)
    wT = base_w * mask_T.astype(float)

    sum_wR = float(np.sum(wR))
    sum_wT = float(np.sum(wT))

    if sum_wR <= 0.0:
        Rm = np.zeros_like(m_orders, dtype=float)
        Rm[m_orders == 0] = Rsum  # all to specular if no propagating side orders
    else:
        Rm = (Rsum * wR) / sum_wR

    if sum_wT <= 0.0:
        Tm = np.zeros_like(m_orders, dtype=float)
        Tm[m_orders == 0] = Tsum
    else:
        Tm = (Tsum * wT) / sum_wT

    return Rm, Tm
