#"""
#Streamlit UI — wired to the mock solver via orchestration.session ports.
#Run with:  streamlit run ui_streamlit/app.py
#"""
from __future__ import annotations

import numpy as np
import streamlit as st

from rcwa_app.adapters.solver_mock.engine import MockSolverEngine
from rcwa_app.orchestration.session import (
    build_sweep_request,
    init_session,
    update_geometry,
    update_illumination,
)
from rcwa_app.plotting_plotly.presenter import PlotPresenterPlotly

# --- App bootstrap ---
if "session" not in st.session_state:
    st.session_state.session = init_session()

engine = MockSolverEngine()
presenter = PlotPresenterPlotly()

st.set_page_config(page_title="RCWA Emissivity — Modular UI", layout="wide")
st.title("Directional Emissivity — Modular UI (Mock Compute)")

# --- Sidebar: Controls ---
with st.sidebar:
    st.header("Model Preset")
    st.caption("(Presets store/load to be added later)")

    st.header("Geometry")
    Ax = st.slider("Ax (μm)", 0.0, 2.0, st.session_state.session.config.geometry.surface.Ax_um, 0.01)
    Ay = st.slider("Ay (μm)", 0.0, 2.0, st.session_state.session.config.geometry.surface.Ay_um, 0.01)
    Lx = st.slider("Lx (μm)", 0.2, 20.0, st.session_state.session.config.geometry.surface.Lx_um, 0.1)
    Ly = st.slider("Ly (μm)", 0.2, 20.0, st.session_state.session.config.geometry.surface.Ly_um, 0.1)
    phix = st.slider("φx (rad)", -np.pi, np.pi, st.session_state.session.config.geometry.surface.phix_rad, 0.01)
    phiy = st.slider("φy (rad)", -np.pi, np.pi, st.session_state.session.config.geometry.surface.phiy_rad, 0.01)
    rot = st.slider("Rotation (deg)", -90.0, 90.0, st.session_state.session.config.geometry.surface.rot_deg, 0.5)

    # Apply geometry updates
    if st.button("Apply geometry", use_container_width=True):
        st.session_state.session = update_geometry(
            st.session_state.session,
            Ax_um=Ax, Ay_um=Ay, Lx_um=Lx, Ly_um=Ly, phix_rad=phix, phiy_rad=phiy, rot_deg=rot,
        )

    st.divider()
    st.header("Illumination")
    lam_min, lam_max, nlam = st.session_state.session.config.illumination.lambda_um
    th_min, th_max, nth = st.session_state.session.config.illumination.theta_deg
    lam_min = st.number_input("λ min (μm)", value=float(lam_min))
    lam_max = st.number_input("λ max (μm)", value=float(lam_max))
    nlam = st.number_input("λ points", value=int(nlam), min_value=5, max_value=2001, step=1)

    th_min = st.number_input("θ min (deg)", value=float(th_min))
    th_max = st.number_input("θ max (deg)", value=float(th_max))
    nth = st.number_input("θ points", value=int(nth), min_value=3, max_value=721, step=1)

    pol = st.selectbox("Polarization", ["TE", "TM", "UNPOL"], index=["TE","TM","UNPOL"].index(
        st.session_state.session.config.illumination.polarization
    ))

    if st.button("Apply illumination", use_container_width=True):
        st.session_state.session = update_illumination(
            st.session_state.session,
            lambda_span=(float(lam_min), float(lam_max), int(nlam)),
            theta_span=(float(th_min), float(th_max), int(nth)),
            polarization=pol,
        )

    st.divider()
    st.header("Numerics (placeholder)")
    st.caption("Shown for completeness; mock engine ignores these.")
    _N = st.number_input("Fourier orders (odd)", value=st.session_state.session.config.numerics.N_orders, step=2)
    _fac = st.selectbox("Factorization", ["LI_FAST","LI_STRICT","NONE"], index=0)
    _tol = st.number_input("Tolerance", value=st.session_state.session.config.numerics.tol, format="%.1e")

    st.divider()
    run = st.button("Run (mock)", type="primary", use_container_width=True)

# --- Main content ---
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

# KPIs
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Energy residual |1−(R+T+A)|", f"{res.scalars.energy_residual:.2e}")
with col2:
    ds = res.data  # type: ignore
    st.metric("λ points", int(ds.sizes["lambda_um"]))
with col3:
    st.metric("θ points", int(ds.sizes["theta_deg"]))

# Tabs
_tab1, _tab2, _tab3 = st.tabs(["Spectral scans", "Maps", "Orders (sample)"])

with _tab1:
    theta_pick = st.slider("θ for ε(λ) plot", float(ds.theta_deg.min()), float(ds.theta_deg.max()), float(ds.theta_deg.values[len(ds.theta_deg)//2]))
    fig1 = presenter.spectral_plot(res, theta_pick)
    st.plotly_chart(fig1, use_container_width=True)

with _tab2:
    fig2 = presenter.map_eps(res)
    st.plotly_chart(fig2, use_container_width=True)

with _tab3:
    il = st.slider("λ index", 0, int(ds.sizes["lambda_um"]) - 1, int(ds.sizes["lambda_um"])//2)
    it = st.slider("θ index", 0, int(ds.sizes["theta_deg"]) - 1, int(ds.sizes["theta_deg"])//2)
    fig3 = presenter.orders_plot(res, il, it)
    st.plotly_chart(fig3, use_container_width=True)

st.caption("Mock engine results shown. Swap in a real RCWA SolverEngine without changing the UI.")