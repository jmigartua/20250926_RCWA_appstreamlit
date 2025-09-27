# Validation Tab — Implementation Pack

This pack adds a **CSV validation loader**, **RMSE computation on a common λ grid**, **overlay plotting**, a simple **badge**, and a **JSON report** artifact—together with deterministic tests. All code follows your CONTRIBUTING discipline (typed, lintable, mypy‑clean, xarray/pandas friendly, no optional extras required).

> Copy files to the indicated paths. Then run: `ruff check . --fix && black . && mypy . && pytest -q`.

---

## 1) `rcwa_app/adapters/validation_csv/loader.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Literal

import pandas as pd

Pol = Literal["TE", "TM", "UNPOL"]


@dataclass(frozen=True)
class ValidationMeta:
    name: str
    theta_deg: float
    pol: Pol


def read_csv_text(text: str) -> pd.DataFrame:
    """Read CSV text into a pandas DataFrame with minimal assumptions.

    The caller provides column name mapping with `map_columns`.
    """
    return pd.read_csv(StringIO(text))


def map_columns(df: pd.DataFrame, *, lam_col: str, eps_col: str) -> pd.DataFrame:
    """Return a normalized DataFrame with columns: `lambda_um`, `eps`.

    Drops NA rows, sorts by `lambda_um`, clips `eps` to [0, 1].
    """
    if lam_col not in df.columns or eps_col not in df.columns:
        raise KeyError("Provided column names not present in CSV")
    out = df[[lam_col, eps_col]].rename(columns={lam_col: "lambda_um", eps_col: "eps"}).copy()
    out = out.dropna(subset=["lambda_um", "eps"]).reset_index(drop=True)
    out["lambda_um"] = out["lambda_um"].astype(float)
    out["eps"] = out["eps"].astype(float).clip(lower=0.0, upper=1.0)
    out = out.sort_values("lambda_um").reset_index(drop=True)
    return out
```

---

## 2) `rcwa_app/validation/metrics.py`

```python
from __future__ import annotations

from math import sqrt
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

from rcwa_app.exporting.io import dataset_line_at_theta


def _overlap_on_model_grid(lam_model: np.ndarray, lam_ref: np.ndarray, eps_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate reference onto model λ grid over the overlapping range.

    Returns (lam_common, eps_ref_interp). Requires ≥ 3 overlapping points.
    """
    lo = max(float(lam_model.min()), float(lam_ref.min()))
    hi = min(float(lam_model.max()), float(lam_ref.max()))
    mask = (lam_model >= lo) & (lam_model <= hi)
    lam_common = lam_model[mask]
    if lam_common.size < 3:
        raise ValueError("Insufficient spectral overlap between model and reference")
    eps_ref_interp = np.interp(lam_common, lam_ref, eps_ref)
    return lam_common, eps_ref_interp


def rmse_eps_on_common_lambda(ds: xr.Dataset, theta_deg: float, ref: pd.DataFrame) -> float:
    """Compute RMSE between model ε(λ,θ≈selected) and reference ε_ref(λ) on the model grid.

    The model line is taken at the nearest available θ in `ds`.
    The reference is interpolated onto the overlapping model λ grid.
    """
    line = dataset_line_at_theta(ds, theta_deg=theta_deg, var="eps", pol_mode="unpolarized")
    lam_model = line["lambda_um"].to_numpy(dtype=float)
    eps_model = line["eps"].to_numpy(dtype=float)

    lam_ref = ref["lambda_um"].to_numpy(dtype=float)
    eps_ref = ref["eps"].to_numpy(dtype=float)

    lam_common, eps_ref_interp = _overlap_on_model_grid(lam_model, lam_ref, eps_ref)

    # align model values to the masked common region
    mask_common = (lam_model >= lam_common.min()) & (lam_model <= lam_common.max())
    eps_model_common = eps_model[mask_common]

    diff = eps_model_common - eps_ref_interp
    return sqrt(float(np.mean(diff * diff)))


def badge_for_rmse(rmse: float, *, pass_th: float = 0.03, warn_th: float = 0.07) -> str:
    if rmse <= pass_th:
        return "PASS"
    if rmse <= warn_th:
        return "WARN"
    return "FAIL"
```

---

## 3) `rcwa_app/plotting_plotly/presenter.py` — add overlay method

Append this method to your existing `PlotPresenterPlotly` class (no changes to existing methods):

```python
    def spectral_overlay(
        self,
        result: "SolverResult",
        fixed_theta: float,
        ref_df: "pd.DataFrame",
        ref_name: str = "reference",
    ):
        import plotly.graph_objects as go  # local import to keep mypy happy
        import pandas as pd

        base_fig = self.spectral_plot(result, fixed_theta)
        # Add reference trace
        fig = go.Figure(base_fig)
        fig.add_trace(
            go.Scatter(
                x=ref_df["lambda_um"],
                y=ref_df["eps"],
                mode="lines+markers",
                name=f"ε (ref: {ref_name})",
                line=dict(dash="dash"),
            )
        )
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0))
        return fig
```

> Typing note: we import plotly and pandas inside the method to avoid module‑level type stubs; the public surface remains unchanged.

---

## 4) `rcwa_app/reports/validation.py` — JSON record

```python
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict

from typing_extensions import TypedDict


class Thresholds(TypedDict):
    pass_th: float
    warn_th: float


@dataclass(frozen=True)
class ValidationRecord:
    dataset_name: str
    theta_deg: float
    pol: str
    rmse: float
    thresholds: Thresholds
    timestamp: str  # ISO 8601


def make_record(dataset_name: str, theta_deg: float, pol: str, rmse: float, *, pass_th: float, warn_th: float) -> ValidationRecord:
    ts = datetime.now(timezone.utc).isoformat()
    return ValidationRecord(
        dataset_name=dataset_name,
        theta_deg=theta_deg,
        pol=pol,
        rmse=rmse,
        thresholds={"pass_th": pass_th, "warn_th": warn_th},
        timestamp=ts,
    )


def to_json_bytes(rec: ValidationRecord) -> bytes:
    return json.dumps(asdict(rec), indent=2).encode("utf-8")
```

---

## 5) `ui_streamlit/app.py` — add a **Validation** tab

Insert a new tab and sidebar controls (minimal and typed). Keep imports at the very top to satisfy Ruff E402; below we show only the **new pieces** for the Validation tab. You can place the tab right after your existing three tabs:

```python
from rcwa_app.adapters.validation_csv.loader import ValidationMeta, read_csv_text, map_columns, Pol
from rcwa_app.validation.metrics import rmse_eps_on_common_lambda, badge_for_rmse
from rcwa_app.reports.validation import make_record, to_json_bytes
```

Add a **new tab** in your tabs construction:
```python
_tab1, _tab2, _tab3, _tab4 = st.tabs(["Spectral scans", "Maps", "Orders (sample)", "Validation"])  # new _tab4
```

Implementation of the **Validation** tab (drop this inside the `with _tab4:` block):

```python
with _tab4:
    st.subheader("Validation against reference CSV")
    # --- File upload & column mapping ---
    up = st.file_uploader("Upload CSV (reference)", type=["csv"])  # returns UploadedFile or None
    if up is not None:
        text = up.getvalue().decode("utf-8", errors="ignore")
        df_raw = read_csv_text(text)
        st.write("Columns:", list(df_raw.columns))
        c1, c2 = st.columns(2)
        with c1:
            lam_col = st.selectbox("Map: wavelength column", list(df_raw.columns))
        with c2:
            eps_col = st.selectbox("Map: emissivity column", list(df_raw.columns))

        try:
            df_ref = map_columns(df_raw, lam_col=lam_col, eps_col=eps_col)
        except KeyError as e:
            st.error(str(e))
            df_ref = None

        if df_ref is not None:
            # --- Metadata ---
            c3, c4, c5 = st.columns([2,1,1])
            with c3:
                ds_name = st.text_input("Dataset name", value=up.name)
            with c4:
                theta_sel = st.number_input("θ (deg)", value=float(ds.theta_deg.values[len(ds.theta_deg)//2]))
            with c5:
                pol_sel = st.selectbox("pol", ["UNPOL","TE","TM"], index=0)

            # --- Thresholds & RMSE ---
            c6, c7 = st.columns(2)
            with c6:
                pass_th = float(st.number_input("PASS ≤", value=0.03, step=0.005, format="%.3f"))
            with c7:
                warn_th = float(st.number_input("WARN ≤", value=0.07, step=0.005, format="%.3f"))

            rmse = rmse_eps_on_common_lambda(ds, theta_deg=float(theta_sel), ref=df_ref)
            badge = badge_for_rmse(rmse, pass_th=pass_th, warn_th=warn_th)

            # --- Overlay plot ---
            figv = presenter.spectral_overlay(res, float(theta_sel), df_ref, ref_name=ds_name)
            st.plotly_chart(figv, use_container_width=True)

            # --- Badge/KPIs ---
            st.metric("RMSE", f"{rmse:.4f}")
            if badge == "PASS":
                st.success("PASS: within threshold")
            elif badge == "WARN":
                st.warning("WARN: borderline")
            else:
                st.error("FAIL: exceeds threshold")

            # --- Save validation record ---
            rec = make_record(ds_name, float(theta_sel), pol_sel, float(rmse), pass_th=pass_th, warn_th=warn_th)
            st.download_button("Download validation.json", data=to_json_bytes(rec), file_name="validation.json")
    else:
        st.info("Upload a CSV to run validation.")
```

> Notes: we keep everything typed and free of module‑level side effects. The badge logic is simple and editable in the UI; RMSE aligns via interpolation on the overlapping λ range.

---

## 6) Tests

### `tests/test_validation_loader.py`
```python
from __future__ import annotations

from rcwa_app.adapters.validation_csv.loader import read_csv_text, map_columns


def test_loader_maps_and_clips() -> None:
    csv = """w_um,Emiss
3.0, -0.1
4.0, 0.5
5.0, 1.2
"""
    df = read_csv_text(csv)
    norm = map_columns(df, lam_col="w_um", eps_col="Emiss")
    assert list(norm.columns) == ["lambda_um", "eps"]
    # clipped to [0,1] and sorted by wavelength
    assert float(norm.iloc[0]["eps"]) == 0.0
    assert float(norm.iloc[-1]["eps"]) == 1.0
```

### `tests/test_validation_rmse.py`
```python
from __future__ import annotations

import numpy as np
import pandas as pd

from rcwa_app.orchestration.session import default_config
from rcwa_app.domain.models import SweepRequest
from rcwa_app.adapters.solver_mock.engine import MockSolverEngine
from rcwa_app.validation.metrics import rmse_eps_on_common_lambda


def _result_small():
    cfg = default_config()
    req = SweepRequest(
        config=cfg,
        sweep_lambda=True,
        lambda_grid_um=[3.0, 4.0, 5.0, 6.0],
        theta_grid_deg=[0.0, 15.0, 30.0],
    )
    return MockSolverEngine().run(req)


def test_rmse_matches_injected_noise() -> None:
    res = _result_small()
    # Take model line at θ=15°, then inject tiny deterministic noise
    ds = res.data
    lam = ds["lambda_um"].values
    i = int(np.argmin(np.abs(ds["theta_deg"].values - 15.0)))
    y = ds["eps"].isel(theta_deg=i).values
    noise = np.array([0.005, -0.005, 0.005, -0.005], dtype=float)
    y_ref = (y + noise).clip(0.0, 1.0)
    df_ref = pd.DataFrame({"lambda_um": lam, "eps": y_ref})

    rmse = rmse_eps_on_common_lambda(ds, theta_deg=15.0, ref=df_ref)
    # Expected RMSE is sqrt(mean(noise^2)) when no clipping occurs
    expected = float(np.sqrt(np.mean(noise * noise)))
    assert abs(rmse - expected) < 1e-6
```

### `tests/test_presenter_overlay.py`
```python
from __future__ import annotations

import numpy as np
import pandas as pd

from rcwa_app.orchestration.session import default_config
from rcwa_app.domain.models import SweepRequest
from rcwa_app.adapters.solver_mock.engine import MockSolverEngine
from rcwa_app.plotting_plotly.presenter import PlotPresenterPlotly


def test_overlay_has_two_traces() -> None:
    cfg = default_config()
    req = SweepRequest(config=cfg, sweep_lambda=True,
                       lambda_grid_um=[3.0,4.0,5.0], theta_grid_deg=[0.0,15.0,30.0])
    res = MockSolverEngine().run(req)

    lam = res.data["lambda_um"].values
    i = int(np.argmin(np.abs(res.data["theta_deg"].values - 15.0)))
    y = res.data["eps"].isel(theta_deg=i).values
    df_ref = pd.DataFrame({"lambda_um": lam, "eps": y})

    fig = PlotPresenterPlotly().spectral_overlay(res, 15.0, df_ref, ref_name="unit-test")
    # one model trace + one reference trace
    assert len(fig.data) == 2
    names = [tr.name for tr in fig.data]
    assert any(n.lower().startswith("ε (unpolarized)".lower()) for n in names)
    assert any("ref: unit-test" in n for n in names)
```

---

## 7) Acceptance checklist (what to verify locally)
- Upload a CSV with arbitrary headers → map columns → overlay appears.
- Changing θ in the tab updates the overlay and **RMSE** number.
- The badge changes according to thresholds.
- Download `validation.json` includes dataset name, θ, pol, RMSE, thresholds, timestamp.
- `ruff/black/mypy/pytest` all green.

---

## Notes
- We rely on your **mock engine**’s strict dataset contract (`eps`, `Rsum/Tsum/Asum`, optional `Rm/Tm`). The validation logic only needs `eps`.
- We kept **all imports at the top** in each module, and all test functions are annotated `-> None` so mypy stays happy.
- The RMSE computation uses **overlap only**; if overlap is too small, it raises a clear `ValueError` (UI could catch and display a message).

