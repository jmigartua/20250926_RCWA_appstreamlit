from __future__ import annotations

import numpy as np

from rcwa_app.adapters.solver_rcwa.engine import RcwaSolverEngine
from rcwa_app.domain.models import Layer, SweepRequest
from rcwa_app.orchestration.session import default_config


def test_planar_energy_conservation() -> None:
    cfg = default_config()
    # Define a simple non-absorbing stack: Air | SiO2 film | Air
    cfg = cfg.model_copy(
        update={
            "geometry": cfg.geometry.model_copy(
                update={
                    "stack": [
                        Layer(
                            name="film",
                            material_ref="SiO2_1p45",
                            thickness_um=1.0,
                            k_override=None,
                            transparent_cap_depth_um=0.0,
                        ),
                        Layer(
                            name="substrate",
                            material_ref="Air",
                            thickness_um=None,
                            k_override=None,
                            transparent_cap_depth_um=0.0,
                        ),
                    ]
                }
            )
        }
    )

    req = SweepRequest(
        config=cfg,
        sweep_lambda=True,
        lambda_grid_um=[3.0, 4.0, 5.0],
        theta_grid_deg=[0.0, 30.0],
    )

    eng = RcwaSolverEngine()
    res = eng.run(req)
    ds = res.data

    energy = ds["Rsum"] + ds["Tsum"] + ds["Asum"]
    assert float(np.nanmax(np.abs(energy.values - 1.0))) <= 5e-6
    # Orders collapse to m=0
    assert ds["order"].size == 1 and int(ds["order"][0]) == 0
    # Emissivity equals absorption by construction
    assert np.allclose(ds["eps"].values, ds["Asum"].values)
