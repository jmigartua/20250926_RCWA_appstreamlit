# --- standard library / typing ----------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, cast

# --- third-party -----------------------------------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

# --- first-party: presets & reports ----------------------------------------------------
from rcwa_app.adapters.presets_local.store import LocalPresetStore

# --- first-party: registry & domain ports/models ---------------------------------------
from rcwa_app.adapters.registry import list_engines, make_engine

# --- first-party: validation tab --------------------------------------------------------
from rcwa_app.adapters.validation_csv.loader import Pol, map_columns, read_csv_text
from rcwa_app.domain.models import SolverResult, SweepRequest
from rcwa_app.domain.ports import SolverEngine

# --- first-party: exporting and plotting -----------------------------------------------
from rcwa_app.exporting.io import (
    dataset_line_at_theta,
    dataset_long_table,
    figure_to_png_bytes,
    to_csv_bytes,
)

# --- first-party: orchestration & configuration ----------------------------------------
from rcwa_app.orchestration.session import (
    init_session,
    update_geometry,
    update_illumination,
)
from rcwa_app.plotting_plotly.presenter import PlotPresenterPlotly
from rcwa_app.reports.methods import methods_markdown
from rcwa_app.validation.metrics import rmse_eps_on_common_lambda

# --------------------------------------------------------------------------------------
# App bootstrap
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="RCWA App", layout="wide")
st.title("RCWA App — Modular UI (Mock / Planar / RCWA-1D)")

# Session object (typed Pydantic config inside)
if "session" not in st.session_state:
    st.session_state.session = init_session()

session = st.session_state.session
presenter = PlotPresenterPlotly()
presets = LocalPresetStore(Path("docs/presets"))

# --------------------------------------------------------------------------------------
# Sidebar — engine selection & model controls
# --------------------------------------------------------------------------------------
st.sidebar.header("Engine / Model Controls")

# Engine selector (Registry)
engine_name = st.sidebar.selectbox("Engine", list_engines(), index=0)

engine_kwargs: dict[str, object] = {}
if engine_name == "RCWA-1D (rigorous)":
    with st.sidebar.expander("Rigorous mode (developer)", expanded=False):
        mode = st.selectbox("Order distribution mode", ["skeleton", "rigorous-lite"], index=0)
        engine_kwargs["mode"] = mode

engine: SolverEngine = make_engine(engine_name, **engine_kwargs)


# Geometry controls
geom = session.config.geometry
surf = geom.surface
with st.sidebar.expander("Geometry — Surface", expanded=False):
    Ax = st.number_input("Ax (μm)", value=float(getattr(surf, "Ax_um", 0.6)), step=0.1)
    Ay = st.number_input("Ay (μm)", value=float(getattr(surf, "Ay_um", 0.6)), step=0.1)
    Lx = st.number_input("Lx (μm)", value=float(getattr(surf, "Lx_um", 10.0)), step=0.5)
    Ly = st.number_input("Ly (μm)", value=float(getattr(surf, "Ly_um", 10.0)), step=0.5)
    duty = st.slider("Duty", 0.0, 1.0, float(getattr(surf, "duty", 0.5)))
    rot = st.number_input("Rotation (deg)", value=float(getattr(surf, "rot_deg", 0.0)), step=1.0)
    if st.button("Apply geometry", use_container_width=True):
        st.session_state.session = update_geometry(
            session,
            Ax_um=float(Ax),
            Ay_um=float(Ay),
            Lx_um=float(Lx),
            Ly_um=float(Ly),
            duty=float(duty),
            rot_deg=float(rot),
        )
        session = st.session_state.session
        st.success("Geometry updated.", icon="✅")

# Illumination controls (polarization only here)
ill = session.config.illumination
with st.sidebar.expander("Illumination", expanded=False):
    pol_sel = st.selectbox(
        "Polarization",
        ["TE", "TM", "UNPOL"],
        index=["TE", "TM", "UNPOL"].index(getattr(ill, "polarization", "UNPOL")),
    )
    pol: Pol = cast(Pol, pol_sel)
    # get current spans from config (tuples like (lo, hi, n))
    lam_span = cast(tuple[float, float, int], getattr(ill, "lambda_um", (3.0, 6.0, 21)))
    th_span = cast(tuple[float, float, int], getattr(ill, "theta_deg", (0.0, 60.0, 5)))
    if st.button("Apply illumination", use_container_width=True):
        st.session_state.session = update_illumination(
            session,
            polarization=pol,
            lambda_span=lam_span,  # required by your API
            theta_span=th_span,  # required by your API
        )
        session = st.session_state.session
        st.success("Illumination updated.", icon="✅")

# Presets (defensive API usage: delete/list/root optional)
with st.sidebar.expander("Presets", expanded=False):
    name = st.text_input("Preset name", value="my_preset")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Save", use_container_width=True):
            presets.save(name, session.config)
            st.success(f"Preset '{name}' saved.", icon="💾")
    with c2:
        if st.button("Load", use_container_width=True):
            cfg = presets.load(name)
            st.session_state.session.config = cfg
            session = st.session_state.session
            st.success(f"Preset '{name}' loaded.", icon="📥")
    with c3:
        del_fn: Any = getattr(presets, "delete", None) or getattr(presets, "remove", None)
        if st.button("Delete", use_container_width=True):
            if callable(del_fn):
                del_fn(name)
                st.warning(f"Preset '{name}' deleted.", icon="🗑️")
            else:
                st.info("Delete not supported by this PresetStore.")
    # list available names, if supported
    list_fn: Any = getattr(presets, "list_names", None) or getattr(presets, "list", None)
    names: list[str] = list_fn() if callable(list_fn) else []
    st.caption(f"Available: {', '.join(names) or '(none)'}")

# --------------------------------------------------------------------------------------
# Sweep grids (UI)
# --------------------------------------------------------------------------------------
st.subheader("Sweep grids")

lam_min, lam_max = st.columns(2)
with lam_min:
    lam_lo = st.number_input("λ min (μm)", value=float(lam_span[0]), step=0.5)
with lam_max:
    lam_hi = st.number_input("λ max (μm)", value=float(lam_span[1]), step=0.5)
lam_n = st.slider("λ points", min_value=3, max_value=101, value=int(lam_span[2]), step=2)

th_min, th_max = st.columns(2)
with th_min:
    th_lo = st.number_input("θ min (deg)", value=float(th_span[0]), step=5.0)
with th_max:
    th_hi = st.number_input("θ max (deg)", value=float(th_span[1]), step=5.0)
th_n = st.slider("θ points", min_value=1, max_value=25, value=int(th_span[2]))

lambda_grid = np.linspace(float(lam_lo), float(lam_hi), int(lam_n)).tolist()
theta_grid = np.linspace(float(th_lo), float(th_hi), int(th_n)).tolist()

# Build request directly (no build_sweep_request signature mismatch)
req = SweepRequest(
    config=session.config,
    sweep_lambda=True,
    lambda_grid_um=lambda_grid,
    theta_grid_deg=theta_grid,
)

# Run engine (single source of truth used by tabs below)
res: SolverResult = engine.run(req)
ds: xr.Dataset = cast(xr.Dataset, res.data)


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Spectral scans", "Maps", "Orders (sample)", "Validation", "Thermal"]
)


# ---- Tab 1: Spectral scans -----------------------------------------------------------
with tab1:
    st.subheader("ε(λ) spectral scan at a fixed θ")
    theta_pick = float(
        st.selectbox(
            "θ (deg)", options=[f"{t:.2f}" for t in theta_grid], index=len(theta_grid) // 2
        )
    )
    theta_pick = float(theta_pick)

    fig = presenter.spectral_plot(res, theta_pick)
    st.plotly_chart(fig, use_container_width=True)

    # Exports for spectral line
    line_df = dataset_line_at_theta(ds, theta_deg=theta_pick, var="eps", pol_mode="unpolarized")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download CSV (line)",
            data=to_csv_bytes(line_df),
            file_name=f"spectral_line_theta_{theta_pick:.1f}.csv",
            mime="text/csv",
        )
    with c2:
        try:
            png = figure_to_png_bytes(fig)
            st.download_button(
                "Download PNG",
                data=png,
                file_name=f"spectral_line_theta_{theta_pick:.1f}.png",
                mime="image/png",
            )
        except RuntimeError as e:
            st.info(str(e))

# ---- Tab 2: Maps ---------------------------------------------------------------------
with tab2:
    st.subheader("ε(λ, θ) map")
    fmap = presenter.map_eps(res)
    st.plotly_chart(fmap, use_container_width=True)

    # Long-form CSV export (λ, θ, variable columns)
    long_df = dataset_long_table(ds, vars=("eps", "Rsum", "Tsum", "Asum"))
    st.download_button(
        "Download CSV (map – long)",
        data=to_csv_bytes(long_df),
        file_name="map_long.csv",
        mime="text/csv",
    )

# ---- Tab 3: Orders (sample) ----------------------------------------------------------
with tab3:
    st.subheader("Order distributions (example)")
    # Pick a single (λ, θ) point (middle)
    lam_idx = len(lambda_grid) // 2
    th_idx = len(theta_grid) // 2
    lam_sel = float(lambda_grid[lam_idx])
    th_sel = float(theta_grid[th_idx])

    st.write(f"λ = {lam_sel:.3f} μm, θ = {th_sel:.1f}°")
    cols = st.columns(3)
    with cols[0]:
        st.metric("Rsum", f"{float(ds['Rsum'].values[lam_idx, th_idx]):.3f}")
    with cols[1]:
        st.metric("Tsum", f"{float(ds['Tsum'].values[lam_idx, th_idx]):.3f}")
    with cols[2]:
        st.metric("Asum", f"{float(ds['Asum'].values[lam_idx, th_idx]):.3f}")

    st.caption("Energy check: R+T+A should be 1.")
    st.write(
        "Max energy residual:",
        f"{float(np.nanmax(np.abs(ds['Rsum'] + ds['Tsum'] + ds['Asum'] - 1.0))):.3e}",
    )

# ---- Tab 4: Validation ---------------------------------------------------------------
with tab4:
    st.subheader("Validation against reference CSV")
    up = st.file_uploader("Upload CSV (reference)", type=["csv"])
    if up is not None:
        text = up.getvalue().decode("utf-8", errors="ignore")
        df_raw = read_csv_text(text)
        st.write("Columns:", list(df_raw.columns))
        c1, c2 = st.columns(2)
        with c1:
            lam_col = st.selectbox("Map: wavelength column", list(df_raw.columns))
        with c2:
            eps_col = st.selectbox("Map: emissivity column", list(df_raw.columns))

        df_ref: Optional[pd.DataFrame] = None
        try:
            df_ref = map_columns(df_raw, lam_col=lam_col, eps_col=eps_col)
        except KeyError as e:
            st.error(str(e))

        if df_ref is not None:
            c3, c4, c5 = st.columns([2, 1, 1])
            with c3:
                ds_name = st.text_input("Dataset name", value=up.name)
            with c4:
                theta_sel = st.number_input(
                    "θ (deg)",
                    value=float(ds.theta_deg.values[len(ds.theta_deg) // 2]),
                    step=1.0,
                )
            with c5:
                pol_v = st.selectbox("pol (meta)", ["UNPOL", "TE", "TM"], index=0)

            # Thresholds
            c6, c7 = st.columns(2)
            with c6:
                pass_th = float(st.number_input("PASS ≤", value=0.03, step=0.005, format="%.3f"))
            with c7:
                warn_th = float(st.number_input("WARN ≤", value=0.07, step=0.005, format="%.3f"))

            # Overlay and metrics
            figv = presenter.spectral_overlay(res, float(theta_sel), df_ref, ref_name=ds_name)
            st.plotly_chart(figv, use_container_width=True)

            from rcwa_app.validation.metrics import badge_for_rmse, rmse_eps_on_common_lambda

            rmse = rmse_eps_on_common_lambda(ds, theta_deg=float(theta_sel), ref=df_ref)
            badge = badge_for_rmse(rmse, pass_th=pass_th, warn_th=warn_th)
            st.metric("RMSE", f"{rmse:.4f}")
            if badge == "PASS":
                st.success("PASS: within threshold")
            elif badge == "WARN":
                st.warning("WARN: borderline")
            else:
                st.error("FAIL: exceeds threshold")

            # Methods card + JSON validation record
            st.download_button(
                "Download methods.md",
                data=methods_markdown(session.config, engine=engine_name).encode("utf-8"),
                file_name="methods.md",
                mime="text/markdown",
            )

            from rcwa_app.reports.validation import make_record, to_json_bytes

            rec = make_record(
                ds_name, float(theta_sel), pol_v, float(rmse), pass_th=pass_th, warn_th=warn_th
            )
            st.download_button(
                "Download validation.json", data=to_json_bytes(rec), file_name="validation.json"
            )
    else:
        st.info("Upload a CSV to run validation.")

# ---- Tab 5: Thermal -------------------------------------------------------------------
with tab5:
    st.subheader("Planck-weighted emissivity & hemispherical totals")

    colT, colTheta = st.columns(2)
    with colT:
        T_K = float(st.number_input("Temperature (K)", min_value=1.0, value=1000.0, step=50.0))
    with colTheta:
        theta_opt = st.selectbox(
            "Angle for line (deg) — choose 'Hemispherical' for average",
            options=["Hemispherical"] + [f"{t:.1f}" for t in theta_grid],
            index=0,
        )

    from rcwa_app.thermal.planck import hemispherical_average_eps, planck_weighted_band_eps

    if theta_opt == "Hemispherical":
        eps_bar, w = planck_weighted_band_eps(ds, T_K=T_K, theta_deg=None)
        st.metric("ε̄ (Planck-weighted, hemispherical)", f"{eps_bar:.3f}")
        # show hemispherical line too
        hemi = hemispherical_average_eps(ds)
        fig_hemi = presenter.spectral_plot(
            res, fixed_theta=float(theta_grid[len(theta_grid) // 2])
        )  # reuse
        # Replace y with hemispherical curve for display
        import plotly.graph_objects as go  # local import to avoid top-level dependency for mypy

        fig_hemi.data = []  # clear
        fig_hemi.add_trace(
            go.Scatter(x=hemi["lambda_um"].values, y=hemi.values, mode="lines", name="ε̄_hemis(λ)")
        )
        st.plotly_chart(fig_hemi, use_container_width=True)
    else:
        theta_line = float(theta_opt)
        eps_bar, w = planck_weighted_band_eps(ds, T_K=T_K, theta_deg=theta_line)
        st.metric("ε̄ (Planck-weighted, line)", f"{eps_bar:.3f}")
        fig_line = presenter.spectral_plot(res, fixed_theta=theta_line)
        st.plotly_chart(fig_line, use_container_width=True)

    # Export the Planck weights on current λ grid
    import pandas as _pd

    df_w = _pd.DataFrame({"lambda_um": ds["lambda_um"].values, "planck_weight": w})
    st.download_button(
        "Download Planck weights (CSV)",
        data=to_csv_bytes(df_w),
        file_name=f"planck_weights_{int(T_K)}K.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------------
root_disp = getattr(presets, "root", None)
root_str = str(root_disp) if root_disp is not None else "(presets dir)"
st.caption(
    f"Engine: **{engine_name}** · Polarization: **{getattr(session.config.illumination, 'polarization', 'UNPOL')}** · Presets path: `{root_str}`"
)
