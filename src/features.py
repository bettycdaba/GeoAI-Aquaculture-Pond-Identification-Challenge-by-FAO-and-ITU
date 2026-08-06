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

# Everything an aggregate feature table is built from: the 12 raw bands plus
# the 4 derived indices.
AGGREGATE_FEATURE_GROUPS = ALL_BANDS + DERIVED_INDEX_NAMES


def build_aggregate_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a (row, month) wide table into ONE row per location, using only
    position-invariant summary stats over whatever months are actually valid.

    This is deliberately calendar-blind: no absolute month index or window
    start position is used, only counts and distribution shape (mean/std/
    min/max). A 4-month window and a 6-month window produce features on the
    same footing, and the same function works whether a row has 4, 5, 6, or
    (for train pre-augmentation) all 12 months present.

    Input `df` must already have gone through `to_nan` and `compute_month_indices`.
    `ID` is always carried through; `label` and `origin_id` are carried through
    only if present (so this same function works on train, augmented train,
    and test without special-casing).
    """
    out = pd.DataFrame(index=df.index)
    out["ID"] = df["ID"]
    if "label" in df.columns:
        out["label"] = df["label"]
    if "origin_id" in df.columns:
        out["origin_id"] = df["origin_id"]

    for feat in AGGREGATE_FEATURE_GROUPS:
        cols = [f"{feat}_{m}" for m in MONTHS]
        sub = df[cols]
        out[f"{feat}_mean"] = sub.mean(axis=1, skipna=True)
        out[f"{feat}_std"] = sub.std(axis=1, skipna=True)
        out[f"{feat}_min"] = sub.min(axis=1, skipna=True)
        out[f"{feat}_max"] = sub.max(axis=1, skipna=True)

    # Valid-month counts: how much evidence this row actually has, and how
    # much of it survived cloud cover. These are legitimate features (not
    # calendar-anchored) and, per notebook 03's verification, their
    # distribution should already closely match between augmented train and
    # real test.
    s1_indicator_cols = [f"VH_{m}" for m in MONTHS]
    s2_indicator_cols = [f"blue_{m}" for m in MONTHS]
    out["n_valid_s1_months"] = df[s1_indicator_cols].notna().sum(axis=1)
    out["n_valid_s2_months"] = df[s2_indicator_cols].notna().sum(axis=1)
    out["s2_valid_fraction_of_s1"] = (
        out["n_valid_s2_months"] / out["n_valid_s1_months"].replace(0, np.nan)
    )

    return out


def aggregate_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """The full set of model-input columns in a feature table: everything
    except ID/label/origin_id metadata."""
    exclude = {"ID", "label", "origin_id"}
    return [c for c in feature_table.columns if c not in exclude]


# Per notebook 04's adversarial-validation cross-reference: raw `blue` and
# `red` aggregates have weak task signal (label correlation 0.18 and 0.04)
# but meaningful train/test shift signal (adversarial importance 126.0 and
# 71.4) -- a poor trade-off. Their contribution isn't lost: both feed into
# ndwi/mndwi/ndvi, which are kept. This is the curated default feature set
# used for model comparison in notebook 05, so every model sees the same,
# evidence-based feature list.
DROPPED_RAW_BANDS = ["blue", "red"]


def curated_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    """aggregate_feature_columns, minus the raw blue_*/red_* aggregate
    columns (see DROPPED_RAW_BANDS). Index columns derived from those bands
    (ndwi, mndwi, ndvi) are unaffected."""
    full = aggregate_feature_columns(feature_table)
    dropped_prefixes = tuple(f"{b}_" for b in DROPPED_RAW_BANDS)
    return [c for c in full if not c.startswith(dropped_prefixes)]
