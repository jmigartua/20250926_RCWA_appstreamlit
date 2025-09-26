# RCWA Emissivity App — UI‑first, Compute‑agnostic

An interactive, modular application for **directional spectral emissivity** modelling based on **RCWA/Fourier Modal Method**. The architecture is deliberately **UI‑first** and **compute‑agnostic**: the Streamlit interface drives an **orchestrator** that speaks to swappable **ports** (interfaces). Concrete solvers (a deterministic **mock** and, later, a **real RCWA** engine) are plug‑in adapters that comply with the same **data contracts**.

> **Status:** Starter skeleton (mock solver) is functional; drop‑in real RCWA engine next. Contracts and module boundaries are stable.

---

## ✨ Key features
- **Strict modularity (Ports & Adapters):** UI depends only on **interfaces** and **validated models**, not on any numerical backend.
- **Contracts-first:** a versioned `contracts.md` defines shapes, units, and invariants (energy accounting, Kirchhoff equality).
- **Streamlit UI:** geometry, illumination, numerics (placeholders), and plotting tabs (spectral scans, maps, orders) using Plotly.
- **Deterministic mock solver:** shape‑correct, energy‑consistent results for full UX testing before physics integration.
- **Reproducibility:** session state serialized via typed models; export-ready figures and datasets (planned).

---

## 🗂 Repository layout
```
repo-root/
├─ pyproject.toml              # project metadata (PEP 621) + tools config
├─ README.md                   # this file
├─ docs/
│  └─ contracts.md             # data contracts, units, invariants (v1.0.0)
├─ rcwa_app/                   # installable library (do not couple with UI)
│  ├─ __init__.py
│  ├─ __main__.py              # optional CLI: `rcwa-app`
│  ├─ domain/                  # Pydantic models, enums, ports
│  ├─ orchestration/           # session + controller (pure functions)
│  ├─ adapters/
│  │  └─ solver_mock/          # MockSolverEngine (shape-correct surrogate)
│  └─ plotting_plotly/         # PlotPresenterPlotly (Plotly figures)
├─ ui_streamlit/
│  └─ app.py                   # Streamlit UI wired to ports via orchestrator
└─ tests/                      # (optional) unit/contract/e2e tests
```

---

## 🚀 Quick start

### 1) Create & activate a virtual environment
```bash
python -m venv .rcwa
source .rcwa/bin/activate   # Windows: .rcwa\Scripts\activate
python -m pip install -U pip wheel build hatchling
```

### 2) Install in editable mode (with dev tools)
> **zsh users:** quote the extras to avoid globbing.
```bash
python -m pip install -e '.[dev]'
```
If you prefer only runtime deps:
```bash
python -m pip install -e .
```

### 3) Run the app
**Option A (direct):**
```bash
streamlit run ui_streamlit/app.py
```
**Option B (CLI):** if `[project.scripts] rcwa-app` is enabled in `pyproject.toml`:
```bash
rcwa-app
```

---

## 🧭 Design philosophy
- **Hexagonal architecture (Ports & Adapters):** The UI talks to **ports** (`SolverEngine`, `PlotPresenter`, `MaterialDB`, `PresetStore`). Concrete implementations are adapters—swappable without touching UI code.
- **Contracts over conventions:** `docs/contracts.md` specifies **coordinates** (`lambda_um`, `theta_deg`, `pol`, `order`), **variables** (`eps`, `Rsum`, `Tsum`, `Asum`, `Rm`, `Tm`), **bounds** and **invariants** (\(R+T+A=1\), \(\varepsilon=\alpha\)). All engines return an **`xarray.Dataset`** satisfying these contracts.
- **Typed domain models:** Configuration and results are Pydantic v2 models; UI state is updated via **pure reducers** in the orchestrator.

---

## 🔌 Swapping the solver (mock → real RCWA)
1. Implement `SolverEngine.run(self, req: SweepRequest) -> SolverResult` in `rcwa_app/adapters/solver_rcwa/engine.py`.
2. Make sure your result conforms to `docs/contracts.md` (same dims/coords/variables; set `schema_version`).
3. In `ui_streamlit/app.py`, replace the import:
   ```python
   from rcwa_app.adapters.solver_mock.engine import MockSolverEngine as Engine
   # → from rcwa_app.adapters.solver_rcwa.engine import RcwaSolverEngine as Engine
   engine = Engine()
   ```
No other file needs to change.

---

## 📊 What the mock engine returns
- Unpolarized/TE/TM **ε(λ, θ)** with smooth, geometry‑dependent peaks (Gaussian surrogate), clipped to \([0,1]\).
- **Rsum, Tsum, Asum** consistent with `R+T+A=1` (small, nonzero `Tsum`).
- **Order‑resolved** `Rm, Tm` over orders \(m=-2..+2\) via deterministic weights.
- **Energy residual** diagnostic in `SolverResult.scalars`.

> The mock is **not** physically accurate; it exists solely so the full UI/plotting/persistence pipeline can be validated independently.

---

## 🧩 Configuration (domain models)
Core objects live in `rcwa_app/domain/models.py`:
- `ModelConfig` → `GeometryConfig` (two‑sinusoid surface + layered stack), `IlluminationConfig` (λ, θ ranges; ψ; polarization), `NumericsConfig` (orders, factorization, tol), `ThermalPostConfig` (optional), `materials_model`.
- `SweepRequest` → explicit λ/θ grids.
- `SolverResult` → `xarray.Dataset` + diagnostics; `schema_version` guards compatibility.

See `docs/contracts.md` for authoritative shapes/units.

---

## 🧪 Development workflow
**Lint/format/type:**
```bash
ruff check .
black .
mypy rcwa_app
```
**Tests:** (if `tests/` present)
```bash
pytest -q
```

**Common zsh pitfall:**
```bash
python -m pip install -e '.[dev]'   # quote extras!
```

---

## 🗺 Roadmap (phased)
1. **MVP (done):** UI with mock engine; spectral/heatmap/orders plots; energy KPI.
2. **Presets & exports:** JSON presets; CSV/PNG export; “methods card”.
3. **Validation tab:** overlay paper curves; RMSE; pass/fail badges.
4. **Real RCWA engine:** S‑matrix stabilization + Fourier factorization.
5. **Thermal integration:** Planck‑weighted bands; hemispherical totals.
6. **Performance:** caching tiers; optional Numba/JAX/CuPy backends.
7. **Alternate UIs:** HoloViz Panel/Dash front‑ends against the same ports.

---

## 📚 References
1. Moharam, M. G., & Gaylord, T. K. “Rigorous coupled‑wave analysis of planar‑grating diffraction,” *JOSA* (1981).
2. Li, L. “Use of Fourier series in the analysis of discontinuous periodic structures,” *JOSA A* (1996); and “Formulation and comparison of two recursive matrix algorithms for modeling layered diffraction gratings,” *JOSA A* (1997).
3. Lalanne, P., & Morris, G. M. “Highly improved convergence of the coupled‑wave method for TM polarization,” *JOSA A* (1996).
4. Hoyer, S., & Hamman, J. “xarray: N‑D labeled arrays and datasets in Python,” *J. Open Research Software* (2017).
5. Streamlit Documentation; Pydantic v2 Documentation; Plotly Python Documentation (accessed routinely during development).

---

## 📝 License
MIT (see `LICENSE`, to be added).

## 🙌 Acknowledgements
This project architecture was planned to support reproducible research, teaching, and rapid prototyping of emissivity models for laser‑textured surfaces.

