import numpy as np

from rcwa_app.adapters.solver_rcwa.layer1d import _principal_kz, slab_smatrix_order_basis
from rcwa_app.adapters.solver_rcwa.layer_modal import layer_modal_te_smatrix


def test_modal_li_te_equals_slab_uniform() -> None:
    # Geometry / numerics
    M = 4  # orders each side → N = 2M+1
    N = 2 * M + 1
    lambda_um = 5.0
    k0 = 2.0 * np.pi / lambda_um
    period_um = 10.0
    theta_deg = 17.0
    d_um = 0.7

    # Refractive indices
    n_in = 1.0
    n_layer = 1.5  # uniform layer index
    n_out = 1.2

    # Order grid kx (same N for both constructions)
    m = np.arange(-M, M + 1, dtype=np.complex128)  # [-M..+M]
    beta = np.complex128(n_in) * np.complex128(k0) * np.sin(np.deg2rad(theta_deg))
    G = 2.0 * np.pi / period_um
    kx = beta + G * m

    # --- Uniform slab reference in order basis (TE) ---
    S_slab = slab_smatrix_order_basis(
        pol="TE",
        n_in=n_in,
        n_layer=n_layer,
        n_out=n_out,
        k0=float(k0),
        kx=kx,
        thickness_um=d_um,
    )

    # --- Modal (Li TE) construction with W = I and gamma = kz_layer ---
    W = np.eye(N, dtype=np.complex128)
    gamma = _principal_kz(n_layer, float(k0), kx)  # (N,) principal-branch kz in layer

    S_li = layer_modal_te_smatrix(
        W=W,
        gamma=gamma,
        thickness_um=d_um,
        k0=float(k0),
        kx=kx,
        n_in=n_in,
        n_out=n_out,
        boundary_model="li-te",
    )

    # Block-by-block equality
    assert np.allclose(S_li.S11, S_slab.S11, rtol=1e-12, atol=1e-12)
    assert np.allclose(S_li.S12, S_slab.S12, rtol=1e-12, atol=1e-12)
    assert np.allclose(S_li.S21, S_slab.S21, rtol=1e-12, atol=1e-12)
    assert np.allclose(S_li.S22, S_slab.S22, rtol=1e-12, atol=1e-12)

    # Optional: energy sanity on a random incoming order amplitude vector (unit power not required)
    rng = np.random.default_rng(123)
    a = (rng.standard_normal(N) + 1j * rng.standard_normal(N)).astype(np.complex128)
    # Through the uniform slab reference
    b_ref = S_slab.S11 @ a
    t_ref = S_slab.S21 @ a
    # Through the Li TE modal path
    b_li = S_li.S11 @ a
    t_li = S_li.S21 @ a
    assert np.allclose(b_li, b_ref, rtol=1e-12, atol=1e-12)
    assert np.allclose(t_li, t_ref, rtol=1e-12, atol=1e-12)
