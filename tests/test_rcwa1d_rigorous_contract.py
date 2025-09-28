from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.engine_rigorous1d import Rcwa1DRigorousEngine
from rcwa_app.domain.models import SweepRequest
from rcwa_app.orchestration.session import default_config


def test_rigorous1d_contract_and_energy() -> None:
    cfg = default_config()
    cfg = cfg.model_copy(update={"numerics": cfg.numerics.model_copy(update={"N_orders": 5})})
    req = SweepRequest(
        config=cfg, sweep_lambda=True, lambda_grid_um=[4.0, 5.0], theta_grid_deg=[0.0, 30.0]
    )
    ds = Rcwa1DRigorousEngine().run(req).data

    # Shapes & keys
    for coord in ("lambda_um", "theta_deg", "order"):
        assert coord in ds.coords
    for var in ("eps", "Rsum", "Tsum", "Asum", "Rm", "Tm"):
        assert var in ds.data_vars

    # Energy & order sums
    energy = ds["Rsum"] + ds["Tsum"] + ds["Asum"]
    assert float(np.nanmax(np.abs(energy.values - 1.0))) <= 5e-6
    assert np.allclose(ds["Rm"].sum("order").values, ds["Rsum"].values)
    assert np.allclose(ds["Tm"].sum("order").values, ds["Tsum"].values)

    # m=0 carries totals in the skeleton
    m = ds["order"].values.astype(int)
    idx0 = int(np.where(m == 0)[0][0])
    assert np.all(ds["Rm"].isel(order=idx0).values >= 0.0)
    assert np.all(ds["Tm"].isel(order=idx0).values >= 0.0)
