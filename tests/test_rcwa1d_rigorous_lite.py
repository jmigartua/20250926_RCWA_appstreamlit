from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.engine_rigorous1d import Rcwa1DRigorousEngine
from rcwa_app.domain.models import SweepRequest
from rcwa_app.orchestration.session import default_config


def test_rigorous_lite_distributes_orders_and_preserves_totals() -> None:
    cfg = default_config()
    cfg = cfg.model_copy(
        update={
            "geometry": cfg.geometry.model_copy(
                update={"surface": cfg.geometry.surface.model_copy(update={"Lx_um": 8.0})}
            ),
            "numerics": cfg.numerics.model_copy(update={"N_orders": 7}),
        }
    )
    req = SweepRequest(
        config=cfg,
        sweep_lambda=True,
        lambda_grid_um=[4.0, 5.0],
        theta_grid_deg=[0.0, 25.0],  # oblique includes several propagating orders
    )
    ds = Rcwa1DRigorousEngine(mode="rigorous-lite").run(req).data

    # Order sums match totals
    assert np.allclose(ds["Rm"].sum("order").values, ds["Rsum"].values)
    assert np.allclose(ds["Tm"].sum("order").values, ds["Tsum"].values)

    # If there is any outgoing power at (λ,θ), at least one order must be > 0
    for i_lam in range(ds.sizes["lambda_um"]):
        for j_th in range(ds.sizes["theta_deg"]):
            rt = float(ds["Rsum"].values[i_lam, j_th] + ds["Tsum"].values[i_lam, j_th])
            if rt > 0.0:
                nz = np.count_nonzero(
                    (ds["Rm"][:, i_lam, j_th].values + ds["Tm"][:, i_lam, j_th].values) > 0.0
                )
                assert nz >= 1
