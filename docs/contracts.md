# contracts.md — Data Contracts, Units, and Invariants (v1.0.0)

This document specifies the **stable interface** between the UI/orchestrator and any compute or plotting modules. All implementations MUST conform to these contracts. Grids are explicit, coordinates are named, and units are fixed.

## 1. Coordinates and Units

- **Wavelength**: `lambda_um` (micrometres, μm). Strictly increasing 1D coordinate.
- **Polar angle**: `theta_deg` (degrees). Strictly increasing 1D coordinate.
- **Azimuth**: `psi_deg` (degrees). Scalar parameter for conical incidence; not a coordinate in v1.
- **Polarization**: `pol ∈ {"TE","TM"}`. In v1 datasets, results carry both TE and TM along a `pol` coordinate. “Unpolarized” views are presented as arithmetic averages unless otherwise stated.
- **Diffraction order**: `order` (integers, e.g., −M…0…+M). Present for order-resolved reflectance/transmittance.

## 2. Required Data Variables (xarray Dataset)

Every `SolverResult.data` MUST include:

- `eps(lambda_um, theta_deg, pol) ∈ [0,1]` — directional spectral emissivity.
- `Rsum(lambda_um, theta_deg, pol) ∈ [0,1]` — total reflectance (sum over orders).
- `Tsum(lambda_um, theta_deg, pol) ∈ [0,1]` — total transmittance.
- `Asum(lambda_um, theta_deg, pol) ∈ [0,1]` — total absorptance.

If order-resolved data are available:

- `Rm(order, lambda_um, theta_deg, pol) ∈ [0,1]` — reflectance per order.
- `Tm(order, lambda_um, theta_deg, pol) ∈ [0,1]` — transmittance per order.

## 3. Invariants and Tolerances

- **Energy accounting**: `Rsum + Tsum + Asum = 1 ± tol`, with `tol ≤ 1e−6` for real engines; mocks may use `tol ≤ 1e−4`.
- **Kirchhoff (equilibrium)**: `eps == Asum` for matched `(λ, θ, pol)` in thermal equilibrium. For opaque substrates, `Tsum ≈ 0` and `eps ≈ 1 − Rsum`.
- **Bounds**: All radiometric quantities are in `[0,1]` (after clipping if needed).
- **Monotonic coordinates**: `lambda_um` and `theta_deg` strictly increasing; no NaNs in coordinates.
- **Schema**: `SolverResult.schema_version == "1.0.0"` for this contract.

## 4. Inputs (ModelConfig essentials)

- **Geometry**: two-sinusoid surface with amplitudes `Ax_um, Ay_um ≥ 0`, periods `Lx_um, Ly_um > 0`, phases `phix_rad, phiy_rad`, in-plane rotation `rot_deg`, optional duty cycle `0≤duty≤1`. Layered stack: list of `Layer {name, thickness_um|None, material_ref, k_override?, transparent_cap_depth_um?}`.
- **Illumination**: wavelength span `(λ_min, λ_max, npts)`, angle span `(θ_min, θ_max, npts)`, `psi_deg`, `polarization ∈ {TE,TM,UNPOL}`, ambient index `n_ambient`.
- **Numerics**: Fourier orders `N_orders` (odd recommended), factorization mode, S-matrix stabilization flag, tolerance.
- **Thermal**: temperature `T_K?`, hemispherical toggle, optional bands for integration.

## 5. Errors and Diagnostics

- `SolverResult.scalars.energy_residual` reports max `|1−(R+T+A)|` over the grid.
- Implementations SHOULD attach attributes documenting versions, compile flags, and factorization choices.

## 6. Versioning

- Backwards-compatible changes bump the **minor** version. Breaking changes bump the **major**. Adapters MUST validate `schema_version`.

## 7. References (for scientific grounding)

- Moharam, M. G., & Gaylord, T. K. (1981). Rigorous coupled-wave analysis of planar-grating diffraction. *JOSA*.
- Li, L. (1996, 1997). Fourier factorization and crossed-grating formulations. *JOSA A*.
- Lalanne, P., & Morris, G. M. (1996). Highly improved convergence of the coupled-wave method for TM polarization. *JOSA A*.
- Reciprocity/Kirchhoff equality in spectral directional emissivity: standard radiative transfer texts and NIST notes.