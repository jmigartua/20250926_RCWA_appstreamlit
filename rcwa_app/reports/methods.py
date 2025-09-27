from __future__ import annotations

from datetime import datetime
from textwrap import dedent

from rcwa_app.domain.models import ModelConfig


def methods_markdown(cfg: ModelConfig, *, engine: str, energy_residual: float | None = None) -> str:
    g = cfg.geometry.surface
    lam_min, lam_max, nlam = cfg.illumination.lambda_um
    th_min, th_max, nth = cfg.illumination.theta_deg
    md = f"""
    # Methods (Auto‑generated)

    **Engine:** {engine}  
    **Contracts version:** {cfg.version}  
    **Generated:** {datetime.utcnow().isoformat()}Z

    ## Geometry
    Two‑sinusoid surface: Ax={g.Ax_um:.3g} μm, Ay={g.Ay_um:.3g} μm, Lx={g.Lx_um:.3g} μm, Ly={g.Ly_um:.3g} μm,
    φx={g.phix_rad:.3g} rad, φy={g.phiy_rad:.3g} rad, rotation={g.rot_deg:.3g}°.

    ## Stack
    """
    for i, layer in enumerate(cfg.geometry.stack):
        thick = "semi‑infinite" if layer.thickness_um is None else f"{layer.thickness_um:.3g} μm"
        md += f"- L{i}: {layer.name} — {thick}, material={layer.material_ref}\n"

    md += f"""
    
    ## Illumination
    λ∈[{lam_min:.3g},{lam_max:.3g}] μm × {nlam} pts; θ∈[{th_min:.3g},{th_max:.3g}]° × {nth} pts;
    ψ={cfg.illumination.psi_deg:.3g}°; polarization={cfg.illumination.polarization}.

    ## Numerics (requested)
    N_orders={cfg.numerics.N_orders}, factorization={cfg.numerics.factorization}, S‑matrix={cfg.numerics.s_matrix}, tol={cfg.numerics.tol:.1e}.
    """
    if energy_residual is not None:
        md += f"\n**Energy residual (max |1−(R+T+A)| over grid):** {energy_residual:.2e}.\n"
    return dedent(md).strip() + "\n"
