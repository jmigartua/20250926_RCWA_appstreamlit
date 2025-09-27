from __future__ import annotations

from rcwa_app.adapters.validation_csv.loader import map_columns, read_csv_text


def test_loader_maps_and_clips() -> None:
    csv = """w_um,Emiss
3.0, -0.1
4.0, 0.5
5.0, 1.2
"""
    df = read_csv_text(csv)
    norm = map_columns(df, lam_col="w_um", eps_col="Emiss")
    assert list(norm.columns) == ["lambda_um", "eps"]
    # clipped to [0,1] and sorted by wavelength
    assert float(norm.iloc[0]["eps"]) == 0.0
    assert float(norm.iloc[-1]["eps"]) == 1.0
