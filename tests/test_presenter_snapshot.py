from __future__ import annotations
import numpy as np


def test_spectral_plot_snapshot_like(spectral_fig):
    fig = spectral_fig
    # One trace labeled as unpolarized emissivity
    assert len(fig.data) == 1
    assert fig.data[0].name.lower().startswith("ε (unpolarized)".lower())
    # Axis titles are stable by contract
    assert fig.layout.xaxis.title.text == "Wavelength λ (μm)"
    assert fig.layout.yaxis.title.text == "Directional emissivity ε"
    # Title starts with the expected prefix
    assert fig.layout.title.text.startswith("ε(λ) at θ≈")
    # Structural snapshot: vector lengths and edge values
    x = np.array(fig.data[0].x)
    y = np.array(fig.data[0].y)
    assert x.size == y.size >= 3
    assert x[0] == 3.0 and x[-1] == 5.0  # from small_grids fixture
    # y is bounded in [0,1] per contracts
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0


def test_map_eps_structure(map_fig):
    fig = map_fig
    # Heatmap present with colorbar title ε
    assert len(fig.data) == 1
    heat = fig.data[0]
    assert getattr(heat, "type", "heatmap") == "heatmap"
    assert heat.colorbar.title.text == "ε"
    # z has shape (len(lambda), len(theta)) == (3, 3)
    z = np.array(heat.z)
    assert z.ndim == 2 and z.shape == (3, 3)
    # All values within [0,1]
    assert float(z.min()) >= 0.0 and float(z.max()) <= 1.0