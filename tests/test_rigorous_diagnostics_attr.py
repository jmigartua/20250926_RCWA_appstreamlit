from __future__ import annotations

from rcwa_app.adapters.solver_rcwa.engine_rigorous1d import Rcwa1DRigorousEngine
from rcwa_app.domain.models import SweepRequest
from rcwa_app.orchestration.session import default_config


def test_diagnostics_attribute_present_and_well_formed() -> None:
    cfg = default_config()
    req = SweepRequest(
        config=cfg,
        sweep_lambda=True,
        lambda_grid_um=[4.0, 5.0, 6.0],
        theta_grid_deg=[0.0, 20.0, 40.0],
    )
    ds = Rcwa1DRigorousEngine().run(req).data
    assert "energy_residual_by_N" in ds.attrs
    seq = ds.attrs["energy_residual_by_N"]
    assert isinstance(seq, list) and all(isinstance(t, tuple) and len(t) == 2 for t in seq)
