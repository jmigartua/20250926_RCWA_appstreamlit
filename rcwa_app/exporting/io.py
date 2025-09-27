from __future__ import annotations

import io
from typing import Any, Iterable

import pandas as pd
import xarray as xr


def dataset_long_table(
    ds: xr.Dataset,
    vars: Iterable[str] = ("eps", "Rsum", "Tsum", "Asum"),
    pol_mode: str = "unpolarized",
) -> pd.DataFrame:
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


def dataset_line_at_theta(
    ds: xr.Dataset, theta_deg: float, var: str = "eps", pol_mode: str = "unpolarized"
) -> pd.DataFrame:
    # Select closest theta and return a 2-column table (lambda, value)
    th = ds.coords["theta_deg"].values
    import numpy as np

    i = int(np.argmin(np.abs(th - theta_deg)))
    da = ds[var]
    if pol_mode == "unpolarized" and "pol" in da.dims and da.sizes.get("pol", 1) > 1:
        da = da.mean(dim="pol")
    line = da.isel(theta_deg=i)
    return line.to_dataframe().reset_index()[["lambda_um", var]]


def figure_to_png_bytes(fig: Any) -> bytes:
    """Export a Plotly figure to PNG bytes via Kaleido.
    Raises RuntimeError with a helpful message when Kaleido is not available.
    """
    try:
        return fig.to_image(format="png", engine="kaleido")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Static image export requires the optional 'export' extra: pip install -e '.[export]'"
        ) from e


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
