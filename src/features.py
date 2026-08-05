"""
Per-month derived indices.

These are computed BEFORE any aggregation/masking-window logic, one value per
(row, month), exactly like the raw bands. The point of computing them at this
stage (rather than only on aggregates) is that ratios/differences are more
robust to illumination/calibration drift than raw reflectance/backscatter,
which matters because train and test come from different time periods.

  NDWI   = (green - nir)   / (green + nir)     -- McFeeters water index
  MNDWI  = (green - swir1) / (green + swir1)   -- often better for turbid/
                                                    pond water than NDWI
  NDVI   = (nir - red)     / (nir + red)       -- vegetation context
  VH-VV  = VH - VV (dB)                        -- radar cross-pol difference;
                                                    since bands are already in
                                                    dB (log domain), a dB
                                                    difference is the log-domain
                                                    equivalent of a linear-domain
                                                    ratio, which is the
                                                    standard water/non-water
                                                    radar signature.

All functions expect -9999 already converted to NaN (see `to_nan`), so that
division/subtraction naturally propagates missingness instead of producing
nonsense values from -9999 placeholders.
"""
import numpy as np
import pandas as pd

from src.config import ALL_BANDS, MONTHS, MISSING_VALUE, band_month_col


def to_nan(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Return a copy with -9999 replaced by NaN in the given columns
    (default: all band/month columns)."""
    if cols is None:
        cols = [band_month_col(b, m) for b in ALL_BANDS for m in MONTHS]
    out = df.copy()
    out[cols] = out[cols].replace(MISSING_VALUE, np.nan)
    return out


def _safe_norm_diff(a: pd.Series, b: pd.Series) -> pd.Series:
    """(a - b) / (a + b), returning NaN where the denominator is ~0 to avoid
    division blowups on near-zero reflectance pixels."""
    denom = a + b
    result = (a - b) / denom
    result[denom.abs() < 1e-6] = np.nan
    return result


def compute_month_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a dataframe already passed through `to_nan`, add one column per
    month for each index: ndwi_MM, mndwi_MM, ndvi_MM, vh_minus_vv_MM.
    Returns a new dataframe (input is not mutated).
    """
    out = df.copy()
    for m in MONTHS:
        green = out[band_month_col("green", m)]
        nir = out[band_month_col("nir", m)]
        swir1 = out[band_month_col("swir1", m)]
        red = out[band_month_col("red", m)]
        vh = out[band_month_col("VH", m)]
        vv = out[band_month_col("VV", m)]

        out[f"ndwi_{m}"] = _safe_norm_diff(green, nir)
        out[f"mndwi_{m}"] = _safe_norm_diff(green, swir1)
        out[f"ndvi_{m}"] = _safe_norm_diff(nir, red)
        out[f"vh_minus_vv_{m}"] = vh - vv
    return out


DERIVED_INDEX_NAMES = ["ndwi", "mndwi", "ndvi", "vh_minus_vv"]
