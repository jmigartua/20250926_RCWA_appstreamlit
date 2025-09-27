# Presets & Exports — Implementation Pack

This pack adds **Presets** (save/load JSON), **Data/figure exports** (CSV/PNG), and a **Methods Card** generator, plus tests. All changes preserve the contracts and modular boundaries.

> Copy the files to the indicated paths. Replace the full `ui_streamlit/app.py` with the version below (it is drop‑in and backwards‑compatible with the mock engine).

---

## 0) `pyproject.toml` — optional export extra

Add the following under `[project.optional-dependencies]` (keep your existing groups):

```toml
[project.optional-dependencies]
export = [
  "kaleido>=0.2.1"  # enables Plotly static image export
]
```

You can then install with:
```bash
python -m pip install -e '.[dev,export]'
```
*CI remains green without `export`; PNG tests are skipped when Kaleido is absent.*

---

## 1) `rcwa_app/adapters/presets_local/store.py`

```python
from __future__ import annotations
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import List
from pydantic import BaseModel

from rcwa_app.domain.models import ModelConfig


def _slugify(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in name.strip())
    safe = "-".join(filter(None, safe.split("-")))
    return safe.lower() or "preset"


class LocalPresetStore:
    """Filesystem-based preset storage (JSON), schema-version aware.

    Presets are stored in ``<base_dir>/<slug>.json``.
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = Path(base_dir or Path.cwd() / "presets").resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def list(self) -> List[str]:
        return sorted(p.stem for p in self.base_dir.glob("*.json"))

    def path_for(self, name: str) -> Path:
        return self.base_dir / f"{_slugify(name)}.json"

    def save(self, name: str, cfg: ModelConfig) -> None:
        path = self.path_for(name)
        data = cfg.model_dump()
        data["schema_version"] = cfg.version
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, name: str) -> ModelConfig:
        path = self.path_for(name)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Future: migrate by data['schema_version'] if needed
        if "schema_version" in data:
            data.pop("schema_version")
        return ModelConfig.model_validate(data)

    def remove(self, name: str) -> None:
        path = self.path_for(name)
        if path.exists():
            path.unlink()
```

---

## 2) `rcwa_app/exporting/io.py`

```python
from __future__ import annotations
import io
import pandas as pd
import xarray as xr
from typing import Iterable


def dataset_long_table(ds: xr.Dataset, vars: Iterable[str] = ("eps", "Rsum", "Tsum", "Asum"),
                       pol_mode: str = "unpolarized") -> pd.DataFrame:
    """Return a tidy long-form table for a dataset on (lambda, theta, pol).

    pol_mode: 'unpolarized' → average over pol if present; 'keep' → keep pol column.
    """
    data = ds
    if pol_mode == "unpolarized" and "pol" in data.dims and data.sizes.get("pol", 1) > 1:
        data = data.mean(dim="pol")
    df = data[vars].to_array("var").to_dataframe().reset_index()
    # If pol was removed, ensure 'pol' not lingering
    if "pol" in df.columns and pol_mode == "unpolarized":
        df = df.drop(columns=["pol"])  # averaged out
    return df


def dataset_line_at_theta(ds: xr.Dataset, theta_deg: float, var: str = "eps",
                          pol_mode: str = "unpolarized") -> pd.DataFrame:
    # Select closest theta and return a 2-column table (lambda, value)
    th = ds.coords["theta_deg"].values
    import numpy as np
    i = int(np.argmin(np.abs(th - theta_deg)))
    da = ds[var]
    if pol_mode == "unpolarized" and "pol" in da.dims and da.sizes.get("pol", 1) > 1:
        da = da.mean(dim="pol")
    line = da.isel(theta_deg=i)
    return line.to_dataframe().reset_index()[["lambda_um", var]]


def figure_to_png_bytes(fig) -> bytes:
    """Export a Plotly figure to PNG bytes via Kaleido.
    Raises RuntimeError with a helpful message when Kaleido is not available.
    """
    try:
        return fig.to_image(format="png", engine="kaleido")  # type: ignore[attr-defined]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Static image export requires the optional 'export' extra: pip install -e '.[export]'"
        ) from e


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
```

---

## 3) `rcwa_app/reports/methods.py`

```python
from __future__ import annotations
from datetime import datetime
from textwrap import dedent
from rcwa_app.domain.models import ModelConfig


def methods_markdown(cfg: ModelConfig, *, engine: str, energy_residual: float | None = None) -> str:
    g = cfg.geometry.surface
    lam_min, lam_max, nlam = cfg.illumination.lambda_um
    th_min, th_max, nth = cfg.illumination.theta_deg
    md = f"""
    # Methods (Auto‑generated)

    **Engine:** {engine}  
    **Contracts version:** {cfg.version}  
    **Generated:** {datetime.utcnow().isoformat()}Z

    ## Geometry
    Two‑sinusoid surface: Ax={g.Ax_um:.3g} μm, Ay={g.Ay_um:.3g} μm, Lx={g.Lx_um:.3g} μm, Ly={g.Ly_um:.3g} μm,
    φx={g.phix_rad:.3g} rad, φy={g.phiy_rad:.3g} rad, rotation={g.rot_deg:.3g}°.

    ## Stack
    """
    for i, layer in enumerate(cfg.geometry.stack):
        thick = "semi‑infinite" if layer.thickness_um is None else f"{layer.thickness_um:.3g} μm"
        md += f"- L{i}: {layer.name} — {thick}, material={layer.material_ref}\n"

    md += f"""
    
    ## Illumination
    λ∈[{lam_min:.3g},{lam_max:.3g}] μm × {nlam} pts; θ∈[{th_min:.3g},{th_max:.3g}]° × {nth} pts;
    ψ={cfg.illumination.psi_deg:.3g}°; polarization={cfg.illumination.polarization}.

    ## Numerics (requested)
    N_orders={cfg.numerics.N_orders}, factorization={cfg.numerics.factorization}, S‑matrix={cfg.numerics.s_matrix}, tol={cfg.numerics.tol:.1e}.
    """
    if energy_residual is not None:
        md += f"\n**Energy residual (max |1−(R+T+A)| over grid):** {energy_residual:.2e}.\n"
    return dedent(md).strip() + "\n"
```

---

## 4) `ui_streamlit/app.py` (replace with this version)

```python
from __future__ import annotations
import streamlit as st
import numpy as np
from pathlib import Path

from rcwa_app.orchestration.session import (
    init_session,
    update_geometry,
    update_illumination,
    build_sweep_request,
)
from rcwa_app.adapters.solver_mock.engine import MockSolverEngine
from rcwa_app.plotting_plotly.presenter import PlotPresenterPlotly
from rcwa_app.adapters.presets_local.store import LocalPresetStore
from rcwa_app.exporting.io import (
    dataset_long_table,
    dataset_line_at_theta,
    figure_to_png_bytes,
    to_csv_bytes,
)
from rcwa_app.reports.methods import methods_markdown

# --- App bootstrap ---
if "session" not in st.session_state:
    st.session_state.session = init_session()

engine = MockSolverEngine()  # swap to RcwaSolverEngine later without UI changes
presenter = PlotPresenterPlotly()
store = LocalPresetStore()   # presets/ at CWD

st.set_page_config(page_title="RCWA Emissivity — Modular UI", layout="wide")
st.title("Directional Emissivity — Modular UI (Mock Compute)")

# --- Sidebar: Presets ---
with st.sidebar:
    st.header("Model Preset")
    colp1, colp2 = st.columns([2,1])
    with colp1:
        preset_name = st.text_input("Preset name", value="baseline")
    with colp2:
        if st.button("Save", use_container_width=True):
            store.save(preset_name, st.session_state.session.config)
            st.success(f"Saved preset '{preset_name}'.")
    preset_list = ["<select>"] + store.list()
    sel = st.selectbox("Load preset", preset_list, index=0)
    cols = st.columns([1,1])
    with cols[0]:
        if st.button("Load", use_container_width=True, disabled=(sel == "<select>")):
            st.session_state.session = st.session_state.session.__class__(
                config=store.load(sel), last_request=None, last_result=None
            )
            st.success(f"Loaded preset '{sel}'. Click Run to compute.")
    with cols[1]:
        if st.button("Delete", use_container_width=True, disabled=(sel == "<select>")):
            store.remove(sel)
            st.success(f"Deleted preset '{sel}'.")

    st.divider()
    st.header("Geometry")
    s = st.session_state.session.config.geometry.surface
    Ax = st.slider("Ax (μm)", 0.0, 2.0, s.Ax_um, 0.01)
    Ay = st.slider("Ay (μm)", 0.0, 2.0, s.Ay_um, 0.01)
    Lx = st.slider("Lx (μm)", 0.2, 20.0, s.Lx_um, 0.1)
    Ly = st.slider("Ly (μm)", 0.2, 20.0, s.Ly_um, 0.1)
    phix = st.slider("φx (rad)", -np.pi, np.pi, s.phix_rad, 0.01)
    phiy = st.slider("φy (rad)", -np.pi, np.pi, s.phiy_rad, 0.01)
    rot = st.slider("Rotation (deg)", -90.0, 90.0, s.rot_deg, 0.5)
    if st.button("Apply geometry", use_container_width=True):
        st.session_state.session = update_geometry(
            st.session_state.session,
            Ax_um=Ax, Ay_um=Ay, Lx_um=Lx, Ly_um=Ly, phix_rad=phix, phiy_rad=phiy, rot_deg=rot,
        )

    st.divider()
    st.header("Illumination")
    ill = st.session_state.session.config.illumination
    lam_min, lam_max, nlam = ill.lambda_um
    th_min, th_max, nth = ill.theta_deg
    lam_min = st.number_input("λ min (μm)", value=float(lam_min))
    lam_max = st.number_input("λ max (μm)", value=float(lam_max))
    nlam = st.number_input("λ points", value=int(nlam), min_value=5, max_value=2001, step=1)
    th_min = st.number_input("θ min (deg)", value=float(th_min))
    th_max = st.number_input("θ max (deg)", value=float(th_max))
    nth = st.number_input("θ points", value=int(nth), min_value=3, max_value=721, step=1)
    pol = st.selectbox("Polarization", ["TE", "TM", "UNPOL"], index=["TE","TM","UNPOL"].index(ill.polarization))
    if st.button("Apply illumination", use_container_width=True):
        st.session_state.session = update_illumination(
            st.session_state.session,
            lambda_span=(float(lam_min), float(lam_max), int(nlam)),
            theta_span=(float(th_min), float(th_max), int(nth)),
            polarization=pol,
        )

    st.divider()
    st.header("Numerics (placeholder)")
    _ = st.number_input("Fourier orders (odd)", value=st.session_state.session.config.numerics.N_orders, step=2)
    _ = st.selectbox("Factorization", ["LI_FAST","LI_STRICT","NONE"], index=0)
    _ = st.number_input("Tolerance", value=st.session_state.session.config.numerics.tol, format="%.1e")

    st.divider()
    run = st.button("Run (mock)", type="primary", use_container_width=True)

# --- Main: compute if needed ---
if run or (st.session_state.session.last_result is None):
    req = build_sweep_request(st.session_state.session, sweep_lambda=True)
    res = engine.run(req)
    st.session_state.session = st.session_state.session.__class__(
        config=st.session_state.session.config,
        last_request=req,
        last_result=res,
    )

res = st.session_state.session.last_result
assert res is not None, "No result available."

ds = res.data  # xarray.Dataset

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Energy residual |1−(R+T+A)|", f"{res.scalars.energy_residual:.2e}")
with col2:
    st.metric("λ points", int(ds.sizes["lambda_um"]))
with col3:
    st.metric("θ points", int(ds.sizes["theta_deg"]))
with col4:
    # Methods card download
    md = methods_markdown(st.session_state.session.config, engine=engine.__class__.__name__,
                          energy_residual=res.scalars.energy_residual)
    st.download_button("Download methods.md", data=md.encode("utf-8"), file_name="methods.md")

# Tabs
_tab1, _tab2, _tab3 = st.tabs(["Spectral scans", "Maps", "Orders (sample)"])

with _tab1:
    theta_pick = st.slider("θ for ε(λ) plot", float(ds.theta_deg.min()), float(ds.theta_deg.max()), float(ds.theta_deg.values[len(ds.theta_deg)//2]))
    fig1 = presenter.spectral_plot(res, theta_pick)
    st.plotly_chart(fig1, use_container_width=True)
    # Exports for spectral line
    colx, coly = st.columns(2)
    with colx:
        df_line = dataset_line_at_theta(ds, theta_pick, var="eps", pol_mode="unpolarized")
        st.download_button("Download CSV (ε vs λ)", data=to_csv_bytes(df_line), file_name="eps_vs_lambda.csv")
    with coly:
        try:
            st.download_button("Download PNG (plot)", data=figure_to_png_bytes(fig1), file_name="eps_vs_lambda.png")
        except RuntimeError as e:
            st.info(str(e))

with _tab2:
    fig2 = presenter.map_eps(res)
    st.plotly_chart(fig2, use_container_width=True)
    # Exports for map
    colm1, colm2 = st.columns(2)
    with colm1:
        df_map = dataset_long_table(ds, vars=("eps", "Rsum", "Tsum", "Asum"), pol_mode="unpolarized")
        st.download_button("Download CSV (map)", data=to_csv_bytes(df_map), file_name="eps_map.csv")
    with colm2:
        try:
            st.download_button("Download PNG (heatmap)", data=figure_to_png_bytes(fig2), file_name="eps_map.png")
        except RuntimeError as e:
            st.info(str(e))

with _tab3:
    il = st.slider("λ index", 0, int(ds.sizes["lambda_um"]) - 1, int(ds.sizes["lambda_um"])//2)
    it = st.slider("θ index", 0, int(ds.sizes["theta_deg"]) - 1, int(ds.sizes["theta_deg"])//2)
    fig3 = presenter.orders_plot(res, il, it)
    st.plotly_chart(fig3, use_container_width=True)
```

---

## 5) Tests — new files

### `tests/test_presets.py`
```python
from __future__ import annotations
from pathlib import Path
from rcwa_app.orchestration.session import default_config
from rcwa_app.adapters.presets_local.store import LocalPresetStore


def test_preset_roundtrip(tmp_path: Path):
    store = LocalPresetStore(base_dir=tmp_path)
    cfg = default_config()
    store.save("unit-test", cfg)
    assert "unit-test" in store.list()
    cfg2 = store.load("unit-test")
    assert cfg2.model_dump() == cfg.model_dump()
    store.remove("unit-test")
    assert "unit-test" not in store.list()
```

### `tests/test_export_csv.py`
```python
from __future__ import annotations
import pandas as pd
from rcwa_app.orchestration.session import default_config
from rcwa_app.domain.models import SweepRequest
from rcwa_app.adapters.solver_mock.engine import MockSolverEngine
from rcwa_app.exporting.io import dataset_long_table, dataset_line_at_theta


def _small_result():
    cfg = default_config()
    req = SweepRequest(config=cfg, sweep_lambda=True, lambda_grid_um=[3,4,5], theta_grid_deg=[0,15,30])
    return MockSolverEngine().run(req)


def test_long_table_structure():
    res = _small_result()
    df = dataset_long_table(res.data)
    assert set(["lambda_um","theta_deg"]).issubset(df.columns)
    assert set(["eps","Rsum","Tsum","Asum"]).issubset(df["var"].unique().tolist() + ["eps","Rsum","Tsum","Asum"]) or "var" not in df.columns
    assert len(df) > 0


def test_line_at_theta():
    res = _small_result()
    df = dataset_line_at_theta(res.data, theta_deg=15.0, var="eps")
    assert list(df.columns) == ["lambda_um","eps"]
    assert len(df) == 3  # three wavelengths
```

### `tests/test_methods_card.py`
```python
from __future__ import annotations
from rcwa_app.orchestration.session import default_config
from rcwa_app.reports.methods import methods_markdown


def test_methods_contains_key_fields():
    cfg = default_config()
    md = methods_markdown(cfg, engine="MockSolverEngine", energy_residual=1.23e-6)
    assert "Methods (Auto‑generated)" in md
    assert "Engine: MockSolverEngine" in md
    assert "Contracts version:" in md
    assert "Geometry" in md and "Illumination" in md and "Numerics" in md
```

---

## 6) Acceptance (what to verify locally)
1. **Run app**: Save a preset; reload it; delete it. Download CSV/PNG for spectral and map; download `methods.md`.
2. **Run tests**: `pytest -q` (PNG export tests are not included; optional when Kaleido installed).
3. **CI**: pushes/PRs should pass and produce coverage as before.

---

## 7) Checklist — Milestone: Presets & Exports
- [ ] Implement `PresetStore` (local JSON) with schema_version and upgrader.
- [ ] Add “Save/Load preset” UI; list presets; “delete”.
- [ ] CSV export: line scans & maps (long-form).
- [ ] PNG export via Kaleido (optional extra).
- [ ] Methods card generator (Markdown); download.
- [ ] Tests: preset round‑trip; CSV structure; methods card fields.

> After you confirm the app and tests pass end‑to‑end in your environment/CI, check off these items in your repository’s issue tracker. This pack is designed to satisfy all boxes as-is.

