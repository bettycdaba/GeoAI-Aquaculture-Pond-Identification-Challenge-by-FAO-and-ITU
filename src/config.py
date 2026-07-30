"""
Central configuration for the aquaculture pond detection project.
Import from here instead of re-typing band/month lists in every notebook,
so a change (e.g. adding a band) only has to happen in one place.
"""
from pathlib import Path

# ---- paths -----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TRAIN_RAW = RAW_DIR / "Train.csv"
TEST_RAW = RAW_DIR / "Test.csv"

# ---- data structure ----------------------------------------------------
# Sentinel-1 (radar) bands: cloud-independent, present whenever a month is "active"
S1_BANDS = ["VH", "VV"]

# Sentinel-2 (optical) bands: can be missing within an active month due to cloud cover
S2_BANDS = [
    "blue", "green", "red",
    "re1", "re2", "re3",       # red-edge 1/2/3
    "nir", "nira",             # NIR, narrow NIR
    "swir1", "swir2",
]

ALL_BANDS = S1_BANDS + S2_BANDS  # 12 bands total

# Months are stored as zero-padded strings to match column naming: VH_01, VH_02, ...
MONTHS = [f"{i:02d}" for i in range(1, 13)]

MISSING_VALUE = -9999

# ---- masking-augmentation constants (measured from real Test.csv) -----
# Test windows are 4, 5, or 6 consecutive months, roughly equal thirds.
TEST_WINDOW_SIZES = [4, 5, 6]
TEST_WINDOW_SIZE_WEIGHTS = [1 / 3, 1 / 3, 1 / 3]

# Within an S1-active month, S2 is additionally cloud-masked ~6.2% of the time
# (measured: 320 / 5147 (row, month) pairs in real Test.csv).
S2_DROPOUT_RATE_GIVEN_S1 = 0.062

RANDOM_SEED = 42

def band_month_col(band: str, month: str) -> str:
    """Column name for a given band/month, e.g. band_month_col('VH', '01') -> 'VH_01'."""
    return f"{band}_{month}"
