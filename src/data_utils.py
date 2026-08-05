"""
Reusable I/O and missingness-inspection helpers.
Kept separate from feature engineering / masking logic so notebook 01
(pure exploration) doesn't need to import anything beyond this.
"""
import numpy as np
import pandas as pd

from src.config import (
    ALL_BANDS, MONTHS, MISSING_VALUE, TRAIN_RAW, TEST_RAW, band_month_col,
)


def load_raw_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_RAW)


def load_raw_test() -> pd.DataFrame:
    return pd.read_csv(TEST_RAW)


def _is_missing(series: pd.Series) -> pd.Series:
    """A value counts as missing whether it's still the raw -9999 sentinel
    or has already been converted to NaN — lets these helpers work on raw
    dataframes (straight from CSV) and on cleaned/augmented ones alike."""
    return series.isna() | (series == MISSING_VALUE)


def month_missing_matrix(df: pd.DataFrame, band: str = "VH") -> np.ndarray:
    """
    For a given indicator band (default VH, since S1 is present whenever a
    month is 'active' at all), return an (n_rows, 12) 0/1 matrix where 1
    means that month is missing (either -9999 or NaN) for that row.
    """
    mat = np.zeros((len(df), len(MONTHS)), dtype=int)
    for mi, m in enumerate(MONTHS):
        col = band_month_col(band, m)
        mat[:, mi] = _is_missing(df[col]).astype(int)
    return mat

def per_band_missing_rate(df: pd.DataFrame) -> pd.Series:
    """Fraction of missing (-9999 or NaN) entries per band, averaged across
    all 12 months."""
    rates = {}
    for band in ALL_BANDS:
        cols = [band_month_col(band, m) for m in MONTHS]
        rates[band] = _is_missing(df[cols]).values.mean()
    return pd.Series(rates).sort_values(ascending=False)


def active_window_size(df: pd.DataFrame, band: str = "VH") -> np.ndarray:
    """Number of non-missing months per row, using `band` as the indicator."""
    return (1 - month_missing_matrix(df, band=band)).sum(axis=1)


def s2_dropout_given_s1_rate(df: pd.DataFrame, s2_band: str = "blue") -> float:
    """
    Of the (row, month) pairs where S1 (VH) is present, what fraction have
    the given S2 band missing? This is the real-world cloud-dropout rate
    we replicate during masking augmentation.
    """
    s1_present = 1 - month_missing_matrix(df, band="VH")
    s2_present = 1 - month_missing_matrix(df, band=s2_band)
    s1_on = s1_present == 1
    s2_off_given_s1_on = (s2_present == 0) & s1_on
    return s2_off_given_s1_on.sum() / s1_on.sum()
