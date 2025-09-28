# """
# Streamlit UI — wired to the mock solver via orchestration.session ports.
# Run with:  streamlit run ui_streamlit/app.py
# """
from __future__ import annotations

from typing import Optional, cast

# stdlib / 3p
import numpy as np
import pandas as pd
import streamlit as st

# local imports — keep these here (no code above)
from rcwa_app.adapters.presets_local.store import LocalPresetStore
from rcwa_app.adapters.solver_rcwa.engine import (
    RcwaSolverEngine as MockSolverEngine,  # temporary swap for UI
)
from rcwa_app.adapters.validation_csv.loader import Pol, map_columns, read_csv_text
from rcwa_app.exporting.io import (
    dataset_line_at_theta,
    dataset_long_table,
    figure_to_png_bytes,
    to_csv_bytes,
)
from rcwa_app.orchestration.session import (
    build_sweep_request,
    init_session,
    update_geometry,
    update_illumination,
)
from rcwa_app.plotting_plotly.presenter import PlotPresenterPlotly
from rcwa_app.reports.methods import methods_markdown
from rcwa_app.reports.validation import make_record, to_json_bytes
from rcwa_app.validation.metrics import badge_for_rmse, rmse_eps_on_common_lambda

# type alias AFTER all imports (ok for Ruff)
# Pol = Literal["TE", "TM", "UNPOL"]

# --- App bootstrap ---
if "session" not in st.session_state:
    st.session_state.session = init_session()

engine = MockSolverEngine()  # swap to RcwaSolverEngine later without UI changes
presenter = PlotPresenterPlotly()
store = LocalPresetStore()  # presets/ at CWD

st.set_page_config(page_title="RCWA Emissivity — Modular UI", layout="wide")
st.title("Directional Emissivity — Modular UI (Mock Compute)")

# --- Sidebar: Presets ---
with st.sidebar:
    st.header("Model Preset")
    colp1, colp2 = st.columns([2, 1])
    with colp1:
        preset_name = st.text_input("Preset name", value="baseline")
    with colp2:
        if st.button("Save", use_container_width=True):
            store.save(preset_name, st.session_state.session.config)
            st.success(f"Saved preset '{preset_name}'.")
    preset_list = ["<select>"] + store.list()
    sel = st.selectbox("Load preset", preset_list, index=0)
    cols = st.columns([1, 1])
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
            Ax_um=Ax,
            Ay_um=Ay,
            Lx_um=Lx,
            Ly_um=Ly,
            phix_rad=phix,
            phiy_rad=phiy,
            rot_deg=rot,
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
    pol_sel = st.selectbox(
        "Polarization", ["TE", "TM", "UNPOL"], index=["TE", "TM", "UNPOL"].index(ill.polarization)
    )
    pol: Pol = cast(Pol, pol_sel)
    if st.button("Apply illumination", use_container_width=True):
        st.session_state.session = update_illumination(
            st.session_state.session,
            lambda_span=(float(lam_min), float(lam_max), int(nlam)),
            theta_span=(float(th_min), float(th_max), int(nth)),
            polarization=pol,
        )

    st.divider()
    st.header("Numerics (placeholder)")
    _ = st.number_input(
        "Fourier orders (odd)", value=st.session_state.session.config.numerics.N_orders, step=2
    )
    _ = st.selectbox("Factorization", ["LI_FAST", "LI_STRICT", "NONE"], index=0)
    _ = st.number_input(
        "Tolerance", value=st.session_state.session.config.numerics.tol, format="%.1e"
    )

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
    md = methods_markdown(
        st.session_state.session.config,
        engine=engine.__class__.__name__,
        energy_residual=res.scalars.energy_residual,
    )
    st.download_button("Download methods.md", data=md.encode("utf-8"), file_name="methods.md")

# Tabs
_tab1, _tab2, _tab3, _tab4 = st.tabs(
    ["Spectral scans", "Maps", "Orders (sample)", "Validation"]
)  # new _tab4

with _tab1:
    theta_pick = st.slider(
        "θ for ε(λ) plot",
        float(ds.theta_deg.min()),
        float(ds.theta_deg.max()),
        float(ds.theta_deg.values[len(ds.theta_deg) // 2]),
    )
    fig1 = presenter.spectral_plot(res, theta_pick)
    st.plotly_chart(fig1, use_container_width=True)
    # Exports for spectral line
    colx, coly = st.columns(2)
    with colx:
        df_line = dataset_line_at_theta(ds, theta_pick, var="eps", pol_mode="unpolarized")
        st.download_button(
            "Download CSV (ε vs λ)", data=to_csv_bytes(df_line), file_name="eps_vs_lambda.csv"
        )
    with coly:
        try:
            st.download_button(
                "Download PNG (plot)", data=figure_to_png_bytes(fig1), file_name="eps_vs_lambda.png"
            )
        except RuntimeError as e:
            st.info(str(e))

with _tab2:
    fig2 = presenter.map_eps(res)
    st.plotly_chart(fig2, use_container_width=True)
    # Exports for map
    colm1, colm2 = st.columns(2)
    with colm1:
        df_map = dataset_long_table(
            ds, vars=("eps", "Rsum", "Tsum", "Asum"), pol_mode="unpolarized"
        )
        st.download_button("Download CSV (map)", data=to_csv_bytes(df_map), file_name="eps_map.csv")
    with colm2:
        try:
            st.download_button(
                "Download PNG (heatmap)", data=figure_to_png_bytes(fig2), file_name="eps_map.png"
            )
        except RuntimeError as e:
            st.info(str(e))

with _tab3:
    il = st.slider("λ index", 0, int(ds.sizes["lambda_um"]) - 1, int(ds.sizes["lambda_um"]) // 2)
    it = st.slider("θ index", 0, int(ds.sizes["theta_deg"]) - 1, int(ds.sizes["theta_deg"]) // 2)
    fig3 = presenter.orders_plot(res, il, it)
    st.plotly_chart(fig3, use_container_width=True)

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

        df_ref: Optional[pd.DataFrame] = None
        try:
            df_ref = map_columns(df_raw, lam_col=lam_col, eps_col=eps_col)
        except KeyError as e:
            st.error(str(e))
            df_ref = None

        if df_ref is not None:
            # --- Metadata ---
            c3, c4, c5 = st.columns([2, 1, 1])
            with c3:
                ds_name = st.text_input("Dataset name", value=up.name)
            with c4:
                theta_sel = st.number_input(
                    "θ (deg)", value=float(ds.theta_deg.values[len(ds.theta_deg) // 2])
                )
            with c5:
                pol_sel = st.selectbox("pol", ["UNPOL", "TE", "TM"], index=0)

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
            rec = make_record(
                ds_name, float(theta_sel), pol_sel, float(rmse), pass_th=pass_th, warn_th=warn_th
            )
            st.download_button(
                "Download validation.json", data=to_json_bytes(rec), file_name="validation.json"
            )
    else:
        st.info("Upload a CSV to run validation.")
