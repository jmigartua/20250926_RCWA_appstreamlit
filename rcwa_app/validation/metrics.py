from __future__ import annotations

from math import sqrt
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr

from rcwa_app.exporting.io import dataset_line_at_theta


def _overlap_on_model_grid(
    lam_model: np.ndarray, lam_ref: np.ndarray, eps_ref: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
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
