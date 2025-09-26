#"""
#Plotly-based presenter implementing PlotPresenter.
#"""
from __future__ import annotations
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from rcwa_app.domain.ports import PlotPresenter
from rcwa_app.domain.models import SolverResult


class PlotPresenterPlotly(PlotPresenter):
    def _unpolarized(self, da: xr.DataArray) -> xr.DataArray:
        if "pol" in da.dims and da.sizes.get("pol", 1) > 1:
            return da.mean(dim="pol")
        return da

    def spectral_plot(self, result: SolverResult, fixed_theta: float) -> go.Figure:
        ds: xr.Dataset = result.data  # type: ignore
        # Pick closest theta
        thetas = ds.coords["theta_deg"].values
        i = int(np.argmin(np.abs(thetas - fixed_theta)))
        eps_line = self._unpolarized(ds["eps"]).isel(theta_deg=i)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eps_line.coords["lambda_um"].values,
                                 y=eps_line.values,
                                 mode="lines",
                                 name="ε (unpolarized)"))
        fig.update_layout(
            xaxis_title="Wavelength λ (μm)",
            yaxis_title="Directional emissivity ε",
            template="plotly_white",
            title=f"ε(λ) at θ≈{float(thetas[i]):.1f}°",
        )
        return fig

    def map_eps(self, result: SolverResult) -> go.Figure:
        ds: xr.Dataset = result.data  # type: ignore
        eps_map = self._unpolarized(ds["eps"])  # (λ, θ)
        fig = go.Figure(
            data=go.Heatmap(
                x=eps_map.coords["theta_deg"].values,
                y=eps_map.coords["lambda_um"].values,
                z=eps_map.values,
                colorbar=dict(title="ε"),
            )
        )
        fig.update_layout(
            xaxis_title="Polar angle θ (deg)",
            yaxis_title="Wavelength λ (μm)",
            template="plotly_white",
            title="Directional emissivity ε(λ, θ)",
        )
        return fig

    def orders_plot(self, result: SolverResult, i_lambda: int, i_theta: int) -> go.Figure:
        ds: xr.Dataset = result.data  # type: ignore
        if "Rm" not in ds or "Tm" not in ds:
            return go.Figure()
        Rm = self._unpolarized(ds["Rm"]).isel(lambda_um=i_lambda, theta_deg=i_theta)
        Tm = self._unpolarized(ds["Tm"]).isel(lambda_um=i_lambda, theta_deg=i_theta)
        orders = ds.coords["order"].values
        fig = go.Figure()
        fig.add_trace(go.Bar(x=orders, y=Rm.values, name="R_m"))
        fig.add_trace(go.Bar(x=orders, y=Tm.values, name="T_m"))
        fig.update_layout(
            barmode="group",
            xaxis_title="Diffraction order m",
            yaxis_title="Power fraction",
            template="plotly_white",
            title="Order-resolved Rm/Tm",
        )
        return fig