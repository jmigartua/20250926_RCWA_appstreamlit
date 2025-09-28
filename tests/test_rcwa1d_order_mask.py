from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.engine_1d import Rcwa1DGratingEngine
from rcwa_app.domain.models import SweepRequest
from rcwa_app.orchestration.session import default_config


def test_1d_grating_equation_mask() -> None:
    cfg = default_config()
    cfg = cfg.model_copy(
        update={
            "geometry": cfg.geometry.model_copy(
                update={
                    "surface": cfg.geometry.surface.model_copy(update={"Ax_um": 0.4, "Lx_um": 10.0})
                }
            ),
            "numerics": cfg.numerics.model_copy(update={"N_orders": 7}),
        }
    )
    req = SweepRequest(
        config=cfg,
        sweep_lambda=True,
        lambda_grid_um=[4.0],  # λ/Λ = 0.4 ⇒ propagating m = −2..+2? (|m*0.4| ≤ 1)
        theta_grid_deg=[0.0],
    )
    ds = Rcwa1DGratingEngine().run(req).data
    m = ds["order"].values.astype(int)
    allowed = np.where(np.abs(m * 0.4) <= 1.0 + 1e-12)[0]
    forbidden = np.setdiff1d(np.arange(m.size), allowed)

    # Forbidden orders carry ~0 power
    assert np.allclose(ds["Rm"].isel(order=forbidden).values, 0.0)
    assert np.allclose(ds["Tm"].isel(order=forbidden).values, 0.0)

    # Sums equal totals
    assert np.allclose(ds["Rm"].sum("order").values, ds["Rsum"].values)
    assert np.allclose(ds["Tm"].sum("order").values, ds["Tsum"].values)
