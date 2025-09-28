# Where we are (baseline)

* **MVP complete**: Streamlit UI wired through orchestrator → mock `SolverEngine`; Plotly presenter delivers **ε(λ)**, **ε(λ,θ)**, and **orders Rm/Tm**; **energy-residual KPI** visible; packaging, CI, coverage badge, and minimal tests are in place.
* **Contracts** are fixed at v1.0.0 (`docs/contracts.md`): named coordinates, invariants (`R+T+A=1`, `ε=Asum`), and shapes.
* **Repository hygiene**: `pyproject.toml` present, MIT license added, CI runs ruff/black/mypy/pytest with coverage and optional Codecov upload.

# Next steps (phased, modular; no cross-module friction)

Below I keep your roadmap headings and add precise deliverables, acceptance criteria, and test hooks. Each item is packaged so it can be completed *standalone*, with ports and contracts unchanged.

---

## 1) Presets & exports

**Goal.** Persist/restore complete UI sessions; export data/figures reproducibly; auto-generate a “methods card”.

**Deliverables**

1. **`PresetStore` adapter** (`adapters/presets_local/store.py`):

   * `save(name: str, cfg: ModelConfig) -> None`, `load(name) -> ModelConfig`, `list() -> list[str]`.
   * Storage: JSON files under `./presets/`. Include `schema_version` and upgrade shim.
   * UI: “Save preset” / “Load preset” in **Sidebar → Model Preset**.
2. **Data exports**:

   * **CSV**: export selected traces and maps from the Presenter (write `xarray.Dataset.to_dataframe().to_csv()` for line scans; for maps, export a tidy long-form table with columns `(lambda_um, theta_deg, pol, eps, Rsum, Tsum, Asum)`).
   * **PNG/SVG**: presenter method `export_figure(fig, path, format='png'|'svg')`, called by a small download button on each tab.
3. **Methods card** (`reports/methods.py`):

   * Generate Markdown capturing: geometry parameters, materials selection (tabulated/Drude), illumination, numerics, and the exact **hash of inputs**; include contract version and app version.

**Acceptance**

* Round-trip: load( save(cfg) ) reproduces a hash of the `ModelConfig`.
* Exported CSV re-imports to reproduce the plotted curve numerically (within float tolerance).
* Methods card includes version strings and a UUID hash of `ModelConfig`.

**Tests**

* Unit tests for `PresetStore` round-trip.
* Snapshot test: export CSV for a small grid and compare structure and head rows.

---

## 2) Validation tab

**Goal.** Overlay paper/reference curves; compute RMSE; show pass/fail badges.

**Deliverables**

1. **Validation loader** (`adapters/validation_csv/loader.py`):

   * Parse arbitrary CSV with a mapping UI (dropdowns to bind “wavelength column”, “emissivity column”, angle/polarization metadata).
2. **Overlay & statistics**:

   * Presenter overlay trace(s) on ε(λ) plot at a selected θ and pol.
   * RMSE (and/or MAE) computed on the **interpolated common λ grid**.
   * Badge logic: **PASS** if RMSE ≤ ε₀ (e.g., 0.03), **WARN** if `0.03 < RMSE ≤ 0.07`, **FAIL** otherwise (thresholds editable in UI).
3. **Validation session artifact**:

   * Store comparison outcomes into `reports/validation.json` with dataset name, date, θ, pol, RMSE, thresholds.

**Acceptance**

* Changing the θ/pol selection updates both overlay and RMSE live.
* Badge and summary are reproducible via export.

**Tests**

* Deterministic CSV fixture; verify RMSE computation against hand-checked values.
* UI smoke: load CSV → overlay shows; RMSE decreases when mock params are tuned.

---

## 3) Real RCWA engine

**Goal.** Introduce a physically faithful `SolverEngine` under `adapters/solver_rcwa/engine.py`, preserving contracts.

**Deliverables**

1. **Engine** (`RcwaSolverEngine`):

   * S-matrix (enhanced transmittance) recursion for multilayers.
   * **Li/Lalanne** Fourier factorization modes (configurable via `NumericsConfig.factorization`).
   * Conical incidence (ψ), TE/TM; order bookkeeping (m indices).
   * Compute `Rsum`, `Tsum`, `Asum`, `Rm`, `Tm` on the same named coordinate grids.
2. **MaterialDB**:

   * `materials_builtin` enhanced with a few tabulated nk sets (e.g., steel, Cr, SiO₂) and a Drude-Lorentz evaluator; *k-override* and *transparent cap depth* implemented.
3. **Diagnostics**:

   * Max energy residual across grid; per-point residual surface in `data.attrs` (optional).
   * Convergence trace if requested (e.g., a list of `(N_orders, residual)` pairs).

**Acceptance**

* **Contract tests** pass (`R+T+A≈1`, `ε=Asum` within tolerance).
* On a small canonical case (1D lamellar TE), numeric results match literature references qualitatively; TM shows expected convergence improvement under factorization.
* UI switch between **mock** and **rcwa** is a one-line change (or config), with no UI edits.

**Tests**

* Property tests (Hypothesis) over small grids to enforce bounds and energy sum.
* Regression test on a frozen parameter set (store expected `Rsum` slice as a fixture with tolerance).

---

## 4) Thermal integration

**Goal.** Planck-weighted band/total emissivities; hemispherical integration.

**Deliverables**

1. **Planck integrator** (`thermal/planck.py`):

   * `band_emissivity(result, T_K, band=(λ1,λ2), pol_mode='unpolarized') -> float`
   * Use spectral radiance ( M_\lambda(T) ) and normalize by the blackbody band integral.
2. **Hemispherical integration** (`thermal/angles.py`):

   * Gaussian quadrature over θ with weight ( \cos\theta\sin\theta ); optional azimuthal average for isotropic cases.
3. **UI**:

   * “Thermal” panel: set `T_K`, bands; show numeric results with uncertainties (if available).

**Acceptance**

* Integrator numerically stable on coarse grids (warn when upsampling is needed).
* Unit test compares against analytic black surface (ε=1 → hemispherical ε=1).

**Tests**

* Unit tests for Planck integrals over known bands at known temperatures.
* Sanity: setting `Asum=ε=0` yields zero band emissivity.

---

## 5) Performance & caching

**Goal.** Keep the UI responsive on dense (λ,θ) grids.

**Deliverables**

1. **Cache layer** (`infra/cache.py`):

   * Tier 1: material dispersion and Fourier matrices keyed by material+λ grid, geometry hash, and `N_orders`.
   * Tier 2: final `SolverResult` keyed by `ModelConfig` hash (minus sweep grids for partial reuse).
   * Backends: in-memory + optional on-disk (`joblib.Memory`) with size/TTL.
2. **Fast/accurate toggle**:

   * “Preview” mode (coarse grids; reduced `N_orders`); **refine on mouse-up** for sliders.
3. **Acceleration** (optional):

   * **Numba** JIT for inner products and block matrix assembly; keep this behind a feature flag.

**Acceptance**

* Profiling shows >50% reduction in repeated compute time on unchanged substructures.
* No change to public contracts.

**Tests**

* Timing tests on repeated runs; cache hit/miss counters.
* Equivalence tests (cached vs fresh) within float tolerances.

---

## 6) Alternate UIs (later)

**Goal.** Demonstrate the same domain/orchestration layers driving **Panel** or **Dash**.

**Deliverables**

* Minimal Panel app (`ui_panel/app.py`) using `param` and the same presenters.
* Keep identical ports; show parity on spectral/heatmap plots.

**Acceptance**

* No edits to `rcwa_app/` core are required to run the alternate UI.

---

# Cross-cutting non-functional work

* **Error taxonomy & UX**: map domain exceptions to user guidance (“increase N_orders”, “enable factorization”, “wavelength out of dispersion table”).
* **Logging & provenance**: structured logs (solver, N_orders, residual) + result hashes stored in exports.
* **Docs**: extend `contracts.md` with a “numerical pitfalls” appendix (TM convergence; factorization selection; conical incidence order mapping).

---

# Issue seeds (copy directly into GitHub)

**Milestone: Presets & Exports**

* [ ] Implement `PresetStore` (local JSON) with schema_version and upgrader.
* [ ] Add “Save/Load preset” UI; list presets; “duplicate as…”.
* [ ] CSV export: line scans & maps (long-form).
* [ ] PNG/SVG export via presenter helper.
* [ ] Methods card generator (Markdown); download.

**Milestone: Validation Tab**

* [ ] CSV loader with column mapping UI.
* [ ] Overlay on ε(λ); selectable θ/pol; interpolation on common grid.
* [ ] RMSE/MAE computation; PASS/WARN/FAIL thresholds in settings.
* [ ] Export validation report (JSON).

**Milestone: RCWA Engine**

* [ ] `RcwaSolverEngine` (S-matrix; Li/Lalanne factorization; conical incidence).
* [ ] MaterialDB (tabulated + Drude-Lorentz) honoring k-override and transparent cap.
* [ ] Contract tests (energy, bounds, shapes); small regression fixture.
* [ ] Convergence diagnostics surface (optional).

**Milestone: Thermal**

* [ ] Planck band integrals; hemispherical integration via Gaussian quadrature.
* [ ] Thermal panel UI; results table; CSV export.

**Milestone: Performance**

* [ ] Cache layer (tiered; keys; TTL).
* [ ] Preview vs accuracy toggle; refine on mouse-up.
* [ ] Numba feature flag; microbenchmarks.

**Milestone: Alternate UI**

* [ ] Panel prototype with param-mirrored controls.
* [ ] Presenter reuse; parity checks.

**Testing & CI**

* [ ] Extend tests: `orders_plot` structure; contract checks (`R+T+A≈1`, `ε=Asum`).
* [ ] Expand matrix to macOS/windows (optional).
* [ ] Coverage gate (e.g., 80%) in CI (optional).

---

## “Definition of Done” (per milestone)

* **Contracts** unchanged and documented.
* **CI** green on 3.10/3.11 (lint, type, tests, coverage report).
* **User-facing docs** updated (README roadmap + methods card notes).
* **Reproducibility** verified: same inputs produce identical exports & methods card hash.

---

If you make the repository public (or share a tarball), I’ll annotate current files and open PR-ready diffs for the first milestone (Presets & Exports)—including the JSON schema, UI wiring, and tests.
