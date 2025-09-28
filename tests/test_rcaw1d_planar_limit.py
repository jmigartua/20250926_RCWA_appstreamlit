from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.engine_1d import Rcwa1DGratingEngine
from rcwa_app.domain.models import SweepRequest
from rcwa_app.orchestration.session import default_config


def test_1d_planar_limit_m0_only() -> None:
    cfg = default_config()
    # Zero amplitude ⇒ all power in m=0
    cfg = cfg.model_copy(
        update={
            "geometry": cfg.geometry.model_copy(
                update={
                    "surface": cfg.geometry.surface.model_copy(update={"Ax_um": 0.0, "Lx_um": 8.0})
                }
            ),
            "numerics": cfg.numerics.model_copy(update={"N_orders": 5}),
        }
    )
    req = SweepRequest(
        config=cfg,
        sweep_lambda=True,
        lambda_grid_um=[4.0, 5.0],
        theta_grid_deg=[0.0, 20.0],
    )
    ds = Rcwa1DGratingEngine().run(req).data

    # Only m=0 carries power; totals preserved
    m = ds["order"].values.astype(int)
    idx0 = int(np.where(m == 0)[0][0])

    # Sums equal planar totals (R can be 0 in Air|Air)
    #    import numpy as np
    assert np.allclose(ds["Rm"].sum("order").values, ds["Rsum"].values)
    assert np.allclose(ds["Tm"].sum("order").values, ds["Tsum"].values)
    assert np.all(ds["Rm"].isel(order=idx0).values >= 0.0)
    assert np.all(ds["Tm"].isel(order=idx0).values >= 0.0)

    # side orders ~ 0
    side = np.delete(np.arange(m.size), idx0)
    assert np.allclose(ds["Rm"].isel(order=side).values, 0.0)
    assert np.allclose(ds["Tm"].isel(order=side).values, 0.0)
