from __future__ import annotations

from rcwa_app.adapters.solver_rcwa.engine import RcwaSolverEngine
from rcwa_app.domain.models import SweepRequest
from rcwa_app.orchestration.session import default_config


def test_engine_contract_shape() -> None:
    cfg = default_config()
    req = SweepRequest(
        config=cfg,
        sweep_lambda=True,
        lambda_grid_um=[4.0, 5.0, 6.0],
        theta_grid_deg=[0.0, 15.0],
    )
    ds = RcwaSolverEngine().run(req).data

    for coord in ("lambda_um", "theta_deg"):
        assert coord in ds.coords and coord in ds.dims

    for var in ("eps", "Rsum", "Tsum", "Asum", "Rm", "Tm"):
        assert var in ds.data_vars

    # Rm/Tm sum to totals
    import numpy as np

    assert np.allclose(ds["Rm"].sum(dim="order").values, ds["Rsum"].values)
    assert np.allclose(ds["Tm"].sum(dim="order").values, ds["Tsum"].values)
