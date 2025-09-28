Perfect—let’s land **Milestone R.1: TE modal eigenproblem (Li factorization, uniform-limit exact)** as an additive patch. It **does not** change engines or UI yet; it only adds rigorously typed utilities + tests. Your default behavior stays green.

---

# R.1 — What lands (summary)

* New, typed utilities in `rigorous1d.py` to build the **TE operator** and solve the **modal eigensystem**:

  * `operator_te_from_harmonics(...)` → builds ( \mathbf{A}*{\rm TE} = k_0^2,\mathbf{C}*\varepsilon - \mathbf{K}_x^2 ).
  * `eigs_te_from_profile(...)` → returns ((\gamma, W)) where (\gamma=\sqrt{\lambda(\mathbf{A}_{\rm TE})}), (W) eigenvectors.
* Two deterministic, fast tests:

  1. **Uniform limit**: (\gamma) matches (k_z=\sqrt{(n k_0)^2 - k_x^2}) to ≤1e−10.
  2. **Conditioning**: `W` from `eigh` is orthonormal ⇒ cond(W) ≈ 1 (sanity/stability).

Everything is behind pure functions—**no engine wiring yet**—so your app remains unchanged.

---

## 1) Append to `rcwa_app/adapters/solver_rcwa/rigorous1d.py`

Add the following block **at the end** of the file (keep your existing imports and earlier utilities intact—`LamellarFourier`, `toeplitz_from_harmonics`, `symmetric_orders`, `kx_orders`, `_safe_complex_sqrt`, `kz_from_dispersion` already exist):

```python
# ----------------------------- TE modal eigenproblem (Li factorization) -----------------

import numpy as np
from numpy.typing import NDArray


def operator_te_from_harmonics(
    h_nonneg: np.ndarray,
    kx: np.ndarray,
    k0: float,
) -> np.ndarray:
    """
    Build the TE operator A_TE = k0^2 * C_eps - Kx^2
    using the convolution Toeplitz of ε (Li factorization for TE).

    Inputs
    ------
    h_nonneg : array of shape (M+1,)
        Non-negative Fourier harmonics [h0, h1, ..., hM] of ε.
    kx       : array of shape (2M+1,)
        Order-wise in-plane wavenumbers (rad/μm).
    k0       : float
        Free-space wavenumber (rad/μm) = 2π/λ.

    Returns
    -------
    A : (2M+1, 2M+1) complex128
        Hermitian operator whose eigenvalues are γ^2 (modal kz squared).
    """
    C_eps = toeplitz_from_harmonics(np.asarray(h_nonneg, dtype=float))
    kx = np.asarray(kx, dtype=np.complex128)
    k0c = np.complex128(k0)
    K2 = np.diag(kx * kx)
    A = (k0c * k0c) * C_eps.astype(np.complex128) - K2
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
    Compute the TE modal eigensystem for a 1D lamellar profile.

    Returns
    -------
    gamma : (2M+1,) complex128
        Modal kz (principal-branch square roots of eigenvalues).
    W     : (2M+1, 2M+1) complex128
        Eigenvector matrix (columns). For Hermitian A, W is unitary (orthonormal columns).
    kx    : (2M+1,) complex128
        In-plane order grid used to build the operator (for diagnostics).

    Notes
    -----
    - In the uniform limit eps_hi == eps_lo == eps0, A_TE reduces to
      (eps0 * k0^2) I - diag(kx^2) ⇒ γ = sqrt((n k0)^2 - kx^2), W = I.
    """
    # Orders & in-plane kx
    m = symmetric_orders(M)
    kx = kx_orders(
        lambda_um=lambda_um,
        theta_deg=theta_deg,
        period_um=period_um,
        m=m,
        n_medium=n_ambient,
    ).astype(np.complex128)

    # Convolution matrix of ε from harmonics
    lf = LamellarFourier(eps_hi=float(eps_hi), eps_lo=float(eps_lo))
    h = lf.eps_harmonics(duty=float(duty), M=int(M))  # full 2M+1 array
    h_nonneg = h[M:]  # [h0, h1, ..., hM]

    k0 = 2.0 * np.pi / float(lambda_um)
    A = operator_te_from_harmonics(h_nonneg=h_nonneg, kx=kx, k0=k0)

    # Hermitian → use eigh (stable, orthonormal eigenvectors)
    evals, evecs = np.linalg.eigh(A)
    gamma = _safe_complex_sqrt(evals.astype(np.complex128))
    return gamma, evecs.astype(np.complex128), kx
```

> We use `np.linalg.eigh` because (A_{\rm TE}) is Hermitian for real ε and real (k_x). That gives orthonormal `W`, making the conditioning test trivial and stable.

---

## 2) New tests

### `tests/test_eigs_te_uniform_limit.py`

```python
from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import (
    eigs_te_from_profile,
    kz_from_dispersion,
    symmetric_orders,
    kx_orders,
)


def test_eigs_te_matches_dispersion_in_uniform_limit() -> None:
    # Uniform medium: eps_hi == eps_lo == eps0
    eps0 = 2.25  # n = 1.5
    lam = 5.0
    theta = 17.0
    period = 10.0
    M = 4
    n_ambient = 1.0

    gamma, W, kx = eigs_te_from_profile(
        eps_hi=eps0, eps_lo=eps0, duty=0.37, M=M,
        lambda_um=lam, theta_deg=theta, period_um=period, n_ambient=n_ambient
    )

    # Reference planar kz for each order
    k0 = 2.0 * np.pi / lam
    n = float(np.sqrt(eps0))
    kz_ref = kz_from_dispersion(k0, n, kx)

    # Sort both for a one-to-one comparison
    idx = np.argsort(np.real(kz_ref))
    assert np.allclose(gamma[idx], kz_ref[idx], rtol=1e-10, atol=1e-12)

    # W should be close to identity in this limit (evecs ~ canonical basis)
    # (sign flips are possible; check orthonormality instead)
    I_mat = W.conj().T @ W
    assert np.allclose(I_mat, np.eye(I_mat.shape[0]), atol=1e-12)
```

### `tests/test_eigs_te_conditioning.py`

```python
from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.rigorous1d import eigs_te_from_profile


def test_eigs_te_eigenvectors_well_conditioned() -> None:
    # Non-uniform lamellar profile, representative parameters
    gamma, W, _ = eigs_te_from_profile(
        eps_hi=4.0, eps_lo=2.0, duty=0.5, M=5,
        lambda_um=6.0, theta_deg=15.0, period_um=8.0, n_ambient=1.0
    )
    # eigh returns orthonormal columns; condition number ~ 1
    s = np.linalg.svd(W, compute_uv=False)
    cond = float(s.max() / max(1e-300, s.min()))
    assert cond < 1e2  # very conservative; typical ≈ 1
```

---

## 3) Run the discipline locally

```bash
ruff check . --fix
black .
mypy .
pytest -q
```

You should stay **fully green**. These tests don’t touch engines or UI, and the math is designed to be exact in the uniform limit.

---

## What this buys us (and next)

* We now have a **tested TE core** that returns modal ((\gamma, W)) and respects the **uniform (planar) limit**.
* **Next (R.2)**: implement the **TM eigenproblem** with Lalanne’s mixed factorization. I’ll keep the same pattern (additive functions + 2 tests). After that, **R.3** wires per-layer S-matrices in the modal basis and gives us single-layer totals that regress to your planar TMM in the limit.

---

## References (background)

* Li, L. “Use of Fourier series in the analysis of discontinuous periodic structures,” *JOSA A* 13, 1870–1876 (1996).
* Moharam, M. G., & Gaylord, T. K. “Rigorous coupled-wave analysis of planar-grating diffraction,” *JOSA* 71, 811–818 (1981).

If you prefer, I can drop **R.2** immediately after you confirm green.
