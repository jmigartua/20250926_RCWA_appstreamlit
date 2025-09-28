from __future__ import annotations

from typing import Dict

# Minimal builtin dispersion: constants or simple Cauchy-like fits (placeholder)
# All values are for wavelength in micrometers (μm). Return complex refractive index n_complex.


class MaterialDB:
    def __init__(self) -> None:
        # Constant-index examples (non-absorbing unless noted)
        self._const: Dict[str, complex] = {
            "Air": 1.0 + 0.0j,
            "SiO2_1p45": 1.45 + 0.0j,  # fused silica approx
            "Al2O3_1p76": 1.76 + 0.0j,  # sapphire ordinary approx
            "Steel_k0p5": 3.0 + 0.5j,  # toy absorbing metal (illustrative)
        }

    def n_of_lambda(self, ref: str, lambda_um: float) -> complex:
        # Extend later with tabulated nk/Drude; for now constants above
        if ref in self._const:
            return self._const[ref]
        # Default to Air when unknown (explicit is better, but defensive)
        return 1.0 + 0.0j
