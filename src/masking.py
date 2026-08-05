"""
Masking augmentation.

Train is fully populated (12/12 months, every band). Test only ever exposes a
contiguous 4-6 month window per row, and even within that window some S2
months are cloud-dropped while S1 survives (see notebook 01 for the measured
rates, mirrored in src/config.py).

If we train on clean train rows and evaluate on masked test rows, the model
has never been asked to make a decision from partial evidence — it's a
different, harder task it was never shown. This module manufactures masked
*copies* of train so the model is trained on the same kind of partial view
it will actually see at test time.

IMPORTANT for downstream use (notebook 05): every augmented copy carries an
`origin_id` column equal to the original row's ID. When doing cross-validation,
you MUST group folds by `origin_id`, not split by row — otherwise two masked
views of the *same* physical location could land in different folds (one in
train, one in validation), which leaks information and makes CV look better
than it will actually generalize.
"""
import numpy as np
import pandas as pd

from src.config import (
    ALL_BANDS, S1_BANDS, S2_BANDS, MONTHS,
    TEST_WINDOW_SIZES, TEST_WINDOW_SIZE_WEIGHTS,
    S2_DROPOUT_RATE_GIVEN_S1, RANDOM_SEED, band_month_col,
)
from src.features import to_nan


def _sample_window(rng: np.random.Generator, n_months: int = 12) -> tuple[int, int]:
    """Sample a contiguous (start_idx, length) window, start-indexed at 0,
    matching the measured test window-size distribution. No wraparound across
    the year boundary — test windows are observed as simple consecutive runs,
    so we don't manufacture a Dec-Jan-style wrap unless evidence says test
    actually has one."""
    length = rng.choice(TEST_WINDOW_SIZES, p=TEST_WINDOW_SIZE_WEIGHTS)
    start = rng.integers(0, n_months - length + 1)
    return int(start), int(length)


def _mask_single_row(row: pd.Series, rng: np.random.Generator) -> pd.Series:
    """Return a copy of one fully-populated row with:
    1. A random contiguous window of months kept, everything else set to NaN.
    2. Within the kept window, S2 bands (not S1) additionally dropped at
       S2_DROPOUT_RATE_GIVEN_S1 per month, independently.
    """
    out = row.copy()
    start, length = _sample_window(rng)
    kept_months = set(MONTHS[start:start + length])

    for m in MONTHS:
        if m not in kept_months:
            # fully masked month: every band goes to NaN
            for b in ALL_BANDS:
                out[band_month_col(b, m)] = np.nan
        else:
            # active month: S1 always survives; S2 has a chance of cloud dropout
            if rng.random() < S2_DROPOUT_RATE_GIVEN_S1:
                for b in S2_BANDS:
                    out[band_month_col(b, m)] = np.nan
    out["window_start_month"] = MONTHS[start]
    out["window_length"] = length
    return out


def build_augmented_train(
    train_df: pd.DataFrame,
    n_augments_per_row: int = 10,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Produce `n_augments_per_row` masked variants of every row in train_df.
    Input train_df must already be -9999-free (run `to_nan` first) and fully
    populated (as raw Train.csv is).

    Returns a long dataframe: n_augments_per_row * len(train_df) rows, each
    tagged with `origin_id` (the original ID, for grouped CV) and
    `variant_id` (which augmentation copy it is).
    """
    rng = np.random.default_rng(seed)
    variants = []
    for variant_id in range(n_augments_per_row):
        copy_df = train_df.copy()
        copy_df["origin_id"] = copy_df["ID"]
        copy_df["variant_id"] = variant_id
        copy_df["ID"] = copy_df["ID"] + f"_v{variant_id}"
        masked_rows = copy_df.apply(lambda row: _mask_single_row(row, rng), axis=1)
        variants.append(masked_rows)
    return pd.concat(variants, ignore_index=True)
