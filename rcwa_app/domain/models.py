#"""
#Domain models (v1.0.0)
#
#Pydantic v2 models define validated configuration and result containers.
#Results are carried as xarray Datasets with named coordinates.
#"""
from __future__ import annotations
from typing import Literal, Optional, List, Tuple
from pydantic import BaseModel, Field

# --- Basic enums/types ---
Pol = Literal["TE", "TM", "UNPOL"]
FactorizationMode = Literal["LI_FAST", "LI_STRICT", "NONE"]


class TwoSinusoidSurface(BaseModel):
    Ax_um: float = Field(..., ge=0.0, description="Amplitude along x (μm)")
    Ay_um: float = Field(..., ge=0.0, description="Amplitude along y (μm)")
    Lx_um: float = Field(..., gt=0.0, description="Period along x (μm)")
    Ly_um: float = Field(..., gt=0.0, description="Period along y (μm)")
    phix_rad: float = 0.0
    phiy_rad: float = 0.0
    rot_deg: float = 0.0
    duty: Optional[float] = Field(None, ge=0.0, le=1.0)


class Layer(BaseModel):
    name: str
    thickness_um: Optional[float] = Field(None, ge=0.0)  # None → semi-infinite
    material_ref: str
    k_override: Optional[float] = Field(None, ge=0.0)
    transparent_cap_depth_um: Optional[float] = Field(None, ge=0.0)


class GeometryConfig(BaseModel):
    surface: TwoSinusoidSurface
    stack: List[Layer]


class IlluminationConfig(BaseModel):
    lambda_um: Tuple[float, float, int]  # (min, max, npts)
    theta_deg: Tuple[float, float, int]  # (min, max, npts)
    psi_deg: float = 0.0
    polarization: Pol = "TM"
    n_ambient: float = 1.0


class NumericsConfig(BaseModel):
    N_orders: int = Field(15, ge=1)
    factorization: FactorizationMode = "LI_FAST"
    s_matrix: bool = True
    tol: float = Field(1e-6, gt=0.0)


class ThermalPostConfig(BaseModel):
    T_K: Optional[float] = None
    hemispherical: bool = False
    bands_um: List[Tuple[float, float]] = []


class ModelConfig(BaseModel):
    geometry: GeometryConfig
    materials_model: Literal["tabulated", "drude_lorentz"] = "tabulated"
    illumination: IlluminationConfig
    numerics: NumericsConfig
    thermal: ThermalPostConfig = ThermalPostConfig()
    version: str = "1.0.0"


# --- Sweep and results ---
class SweepRequest(BaseModel):
    config: ModelConfig
    sweep_lambda: bool = True  # else sweep theta
    lambda_grid_um: Optional[List[float]] = None
    theta_grid_deg: Optional[List[float]] = None


class SolverScalars(BaseModel):
    energy_residual: float
    notes: str = ""


# SolverResult carries an xarray.Dataset in runtime, not validated here to avoid heavy import.
class SolverResult(BaseModel):
    data: object  # xarray.Dataset expected at runtime
    scalars: SolverScalars
    schema_version: str = "1.0.0"