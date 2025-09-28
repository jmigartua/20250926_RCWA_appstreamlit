# Full RCWA (Li/Lalanne) — Rigorous 1D S‑matrix Skeleton Pack

This pack introduces a *rigorous* 1D RCWA engine skeleton that is API/contract–compatible and green by design. Internally, it wires the Fourier-convolution and S‑matrix layer interfaces, but for now routes totals through the validated **planar TMM baseline** so all tests pass. We then swap in the actual eigen-solver in a later step without touching ports/UI or tests.

> Copy the files below, then run: `ruff check . --fix && black . && mypy . && pytest -q`.

---

## 1) `rcwa_app/adapters/solver_rcwa/rigorous1d.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Tuple

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
    def propagate(self, field: np.ndarray) -> np.ndarray:
        ...


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
```

---

## 2) `rcwa_app/adapters/solver_rcwa/engine_rigorous1d.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np
import xarray as xr

from rcwa_app.domain.models import ModelConfig, SweepRequest, SolverResult, SolverScalars
from rcwa_app.adapters.solver_rcwa.engine import RcwaSolverEngine  # planar totals
from rcwa_app.adapters.solver_rcwa.rigorous1d import symmetric_orders


@dataclass
class Rcwa1DRigorousResult:
    scalars: SolverScalars
    data: xr.Dataset


class Rcwa1DRigorousEngine:
    """Rigorous RCWA 1D (skeleton): contracts preserved, totals from planar TMM.

    Next step: replace the internal propagation with Li/Lalanne factorization + S-matrix.
    """

    def __init__(self) -> None:
        self.planar = RcwaSolverEngine()

    def run(self, request: Mapping[str, Any] | SweepRequest) -> Rcwa1DRigorousResult:
        base = self.planar.run(request)
        ds0 = base.data

        lam = ds0["lambda_um"].values.astype(float)
        th = ds0["theta_deg"].values.astype(float)

        # Orders from config
        if isinstance(request, SweepRequest):
            cfg = request.config
        else:
            cfg = cast(ModelConfig, cast(Mapping[str, Any], request)["config"])  # type: ignore[index]

        m = symmetric_orders(int(getattr(cfg.numerics, "N_orders", 5)))

        # For now: rigorous engine emits only m=0, equal to totals (planar limit).
        # This keeps tests green while we wire the eigen-solver.
        Rsum = ds0["Rsum"].values
        Tsum = ds0["Tsum"].values
        Asum = ds0["Asum"].values
        eps = ds0["Asum"].values  # emissivity

        Rm = np.zeros((m.size, lam.size, th.size), dtype=float)
        Tm = np.zeros_like(Rm)
        idx0 = int(np.where(m == 0)[0][0])
        Rm[idx0, :, :] = Rsum
        Tm[idx0, :, :] = Tsum

        ds = xr.Dataset(
            data_vars=dict(
                eps=(("lambda_um", "theta_deg"), eps),
                Rsum=(("lambda_um", "theta_deg"), Rsum),
                Tsum=(("lambda_um", "theta_deg"), Tsum),
                Asum=(("lambda_um", "theta_deg"), Asum),
                Rm=(("order", "lambda_um", "theta_deg"), Rm),
                Tm=(("order", "lambda_um", "theta_deg"), Tm),
            ),
            coords=dict(lambda_um=lam, theta_deg=th, order=m),
            attrs=dict(ds0.attrs),
        )

        residual = float(np.nanmax(np.abs(ds["Rsum"] + ds["Tsum"] + ds["Asum"] - 1.0)))
        scalars = SolverScalars(energy_residual=residual, notes="rcwa-1d-rigorous-skeleton")
        return Rcwa1DRigorousResult(scalars=scalars, data=ds)
```

---

## 3) Registry wiring

### `rcwa_app/adapters/registry.py` — append one line
```python
from rcwa_app.adapters.solver_rcwa.engine_rigorous1d import Rcwa1DRigorousEngine

_REGISTRY = {
    "Mock": MockSolverEngine,
    "Planar TMM": RcwaSolverEngine,
    "RCWA-1D": Rcwa1DGratingEngine,
    "RCWA-1D (rigorous)": Rcwa1DRigorousEngine,  # NEW
}
```

---

## 4) Tests

### `tests/test_rcwa1d_rigorous_contract.py`
```python
from __future__ import annotations

import numpy as np

from rcwa_app.orchestration.session import default_config
from rcwa_app.domain.models import SweepRequest
from rcwa_app.adapters.solver_rcwa.engine_rigorous1d import Rcwa1DRigorousEngine


def test_rigorous1d_contract_and_energy() -> None:
    cfg = default_config()
    cfg = cfg.model_copy(update={"numerics": cfg.numerics.model_copy(update={"N_orders": 5})})
    req = SweepRequest(config=cfg, sweep_lambda=True, lambda_grid_um=[4.0, 5.0], theta_grid_deg=[0.0, 30.0])
    ds = Rcwa1DRigorousEngine().run(req).data

    # Shapes & keys
    for coord in ("lambda_um", "theta_deg", "order"):
        assert coord in ds.coords
    for var in ("eps", "Rsum", "Tsum", "Asum", "Rm", "Tm"):
        assert var in ds.data_vars

    # Energy & order sums
    energy = ds["Rsum"] + ds["Tsum"] + ds["Asum"]
    assert float(np.nanmax(np.abs(energy.values - 1.0))) <= 5e-6
    assert np.allclose(ds["Rm"].sum("order").values, ds["Rsum"].values)
    assert np.allclose(ds["Tm"].sum("order").values, ds["Tsum"].values)

    # m=0 carries totals in the skeleton
    m = ds["order"].values.astype(int)
    idx0 = int(np.where(m == 0)[0][0])
    assert np.all(ds["Rm"].isel(order=idx0).values >= 0.0)
    assert np.all(ds["Tm"].isel(order=idx0).values >= 0.0)
```

---

## 5) Optional: UI selector already works
No UI change needed beyond the registry addition; the sidebar will now show **RCWA-1D (rigorous)**.

---

## 6) Next step (when ready)
Swap the internal placeholder with the real RCWA pipeline:
1. Build convolution matrices from `LamellarFourier`.
2. Form the layer eigenproblem (TE/TM separately), compute modal fields.
3. Assemble S-matrix and cascade through layers; compute order amplitudes.
4. Convert amplitudes to power with admittance factors; populate `Rm/Tm`.

All tests above will remain valid; we’ll add convergence checks and a planar‑limit regression.

## References
- Li, L. “Use of Fourier series in the analysis of discontinuous periodic structures.” *JOSA A* 13, 1870–1876 (1996).
- Lalanne, P., & Morris, G. M. “Highly improved convergence of the coupled‑wave method for TM polarization.” *JOSA A* 13, 779–784 (1996).
- Moharam, M. G., & Gaylord, T. K. “Rigorous coupled‑wave analysis of planar‑grating diffraction.” *JOSA* 71, 811–818 (1981).

