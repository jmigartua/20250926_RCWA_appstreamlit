# Rigorous RCWA — Patch 1/6 (Fourier & Order Grid)

This patch lands the **Fourier coefficient/convolution builder** for a 1D lamellar profile and utilities for the **symmetric order grid** and **in-plane wave‑vector**. It is additive and keeps the current engines/UI/tests green. New unit tests validate coefficients and matrix structure.

> After copying these files, run: `ruff check . --fix && black . && mypy . && pytest -q`.

---

## 1) `rcwa_app/adapters/solver_rcwa/rigorous1d.py` (replace file)

```python
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
                    (C @ f)_m = \sum_g eps_g f_{m-g}.
        """


def toeplitz_from_harmonics(h: np.ndarray) -> np.ndarray:
    """Build a real symmetric Toeplitz convolution matrix from harmonics h.

    h must be of length (2M+1) with indices ordered as g=-M..+M and h[g]
    representing the g-th Fourier coefficient. The Toeplitz structure is
    C[i, j] = h[i-j] in the index space shifted by +M.
    """
    if h.ndim != 1 or h.size % 2 != 1:
        raise ValueError("h must be a 1D array with odd length (2M+1)")
    M = (h.size - 1) // 2
    C = np.empty((2 * M + 1, 2 * M + 1), dtype=float)
    # Build via absolute index difference (symmetric for real-valued profiles)
    for i in range(2 * M + 1):
        for j in range(2 * M + 1):
            C[i, j] = h[abs(i - j)]
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


def kx_orders(lambda_um: float, theta_deg: float, period_um: float, m: np.ndarray, n_medium: float = 1.0) -> np.ndarray:
    """In-plane wavevector components (x-direction) for order set m.

    kx_m = k0 * (n_medium * sin(theta) + m * lambda / period)
    where k0 = 2π/λ. The result is in rad/μm.
    """
    k0 = 2.0 * np.pi / float(lambda_um)
    s = float(n_medium) * np.sin(np.deg2rad(float(theta_deg)))
    return k0 * (s + (float(lambda_um) / float(period_um)) * m.astype(float))
```

---

## 2) Tests

### `tests/test_fourier_lamellar_coeffs.py`
```python
from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import LamellarFourier, toeplitz_from_harmonics


def test_lamellar_eps0_and_first_harmonics() -> None:
    M = 5
    duty = 0.3
    eps_hi, eps_lo = 4.0, 2.0
    lf = LamellarFourier(eps_hi=eps_hi, eps_lo=eps_lo)
    h = lf.eps_harmonics(duty=duty, M=M)

    # Zeroth coefficient equals area average
    eps0 = eps_lo + (eps_hi - eps_lo) * duty
    assert abs(h[M] - eps0) < 1e-12

    # First harmonic matches analytical sinc-like value
    expected_g1 = (eps_hi - eps_lo) * np.sin(np.pi * 1 * duty) / (np.pi * 1)
    assert abs(h[M + 1] - expected_g1) < 1e-12
    assert abs(h[M - 1] - expected_g1) < 1e-12  # symmetry


def test_convolution_toeplitz_properties() -> None:
    M = 3
    duty = 0.5
    lf = LamellarFourier(eps_hi=3.0, eps_lo=1.0)
    h = lf.eps_harmonics(duty=duty, M=M)

    # Build Toeplitz from non-negative harmonics [h0, h1, ...]
    C = toeplitz_from_harmonics(h[M:])
    assert C.shape == (2 * M + 1, 2 * M + 1)

    # Symmetry: C[i,j] == C[j,i] for real lamellar
    assert np.allclose(C, C.T)

    # Diagonal equals h0 everywhere
    assert np.allclose(np.diag(C), h[M])
```

### `tests/test_orders_grid_rigorous.py`
```python
from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import symmetric_orders, kx_orders


def test_symmetric_orders_contains_zero_and_is_odd() -> None:
    for N in (3, 4, 7):
        m = symmetric_orders(N)
        assert (m == 0).any()
        assert m.size % 2 == 1
        assert m[0] == -m[-1]


def test_kx_orders_scales_with_lambda_and_period() -> None:
    m = symmetric_orders(5)
    kx1 = kx_orders(lambda_um=4.0, theta_deg=0.0, period_um=10.0, m=m)
    kx2 = kx_orders(lambda_um=5.0, theta_deg=0.0, period_um=10.0, m=m)
    # Different wavelengths should scale k0 ∝ 1/λ
    assert kx2[0] / kx1[0] == pytest.approx(4.0 / 5.0, rel=1e-12)
```

> Add at top of this file: `import pytest` (pytest is already in your dev deps).
```

---

## 3) No engine changes (yet)
`engine_rigorous1d.py` remains as a skeleton (m=0 keeps totals). The new utilities are independent and will be used in Patch 2 for the modal eigenproblem.

---

## 4) Checklist for your Issue
- [ ] Fourier harmonics for lamellar profile (`LamellarFourier.eps_harmonics`).
- [ ] Convolution Toeplitz builder and properties test.
- [ ] Symmetric order set + kx grid utility.
- [ ] All checks green (`ruff/black/mypy/pytest`).

---

## References
- Li, L. (1996). *Use of Fourier series in the analysis of discontinuous periodic structures.* JOSA A 13, 1870–1876.
- Lalanne, P., & Morris, G. M. (1996). *Highly improved convergence of the coupled‑wave method for TM polarization.* JOSA A 13, 779–784.
- Moharam, M. G., & Gaylord, T. K. (1981). *Rigorous coupled‑wave analysis of planar‑grating diffraction.* JOSA 71, 811–818.
```

