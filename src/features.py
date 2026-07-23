import pandas as pd
import numpy as np

EPSILON = 1e-10

FEATURE_GROUPS = {
    "VH": [f"VH_{i:02d}" for i in range(1, 13)],
    "VV": [f"VV_{i:02d}" for i in range(1, 13)],
    "blue": [f"blue_{i:02d}" for i in range(1, 13)],
    "green": [f"green_{i:02d}" for i in range(1, 13)],
    "red": [f"red_{i:02d}" for i in range(1, 13)],
    "nir": [f"nir_{i:02d}" for i in range(1, 13)],
    "nira": [f"nira_{i:02d}" for i in range(1, 13)],
    "re1": [f"re1_{i:02d}" for i in range(1, 13)],
    "re2": [f"re2_{i:02d}" for i in range(1, 13)],
    "re3": [f"re3_{i:02d}" for i in range(1, 13)],
    "swir1": [f"swir1_{i:02d}" for i in range(1, 13)],
    "swir2": [f"swir2_{i:02d}" for i in range(1, 13)],
}


def engineer_features(df):

    engineered_df = df.copy()

    # Temporal statistics

    for group, cols in FEATURE_GROUPS.items():

        engineered_df[f"{group}_mean"] = df[cols].mean(axis=1)

        engineered_df[f"{group}_std"] = df[cols].std(axis=1)

        engineered_df[f"{group}_min"] = df[cols].min(axis=1)

        engineered_df[f"{group}_max"] = df[cols].max(axis=1)

        engineered_df[f"{group}_range"] = (
            engineered_df[f"{group}_max"]
            - engineered_df[f"{group}_min"]
        )

    # NDVI

    ndvi_cols = []

    for i in range(1, 13):

        col = f"NDVI_{i:02d}"

        engineered_df[col] = (
            (
                df[f"nir_{i:02d}"]
                - df[f"red_{i:02d}"]
            )
            /
            (
                df[f"nir_{i:02d}"]
                + df[f"red_{i:02d}"]
                + EPSILON
            )
        )

        ndvi_cols.append(col)

    engineered_df["NDVI_mean"] = engineered_df[ndvi_cols].mean(axis=1)

    engineered_df["NDVI_std"] = engineered_df[ndvi_cols].std(axis=1)

    # NDWI

    ndwi_cols = []

    for i in range(1, 13):

        col = f"NDWI_{i:02d}"

        engineered_df[col] = (
            (
                df[f"green_{i:02d}"]
                - df[f"nir_{i:02d}"]
            )
            /
            (
                df[f"green_{i:02d}"]
                + df[f"nir_{i:02d}"]
                + EPSILON
            )
        )

        ndwi_cols.append(col)

    engineered_df["NDWI_mean"] = engineered_df[ndwi_cols].mean(axis=1)

    engineered_df["NDWI_std"] = engineered_df[ndwi_cols].std(axis=1)

    # VH/VV Ratio

    ratio_cols = []

    for i in range(1, 13):

        col = f"VH_VV_ratio_{i:02d}"

        engineered_df[col] = (
            df[f"VH_{i:02d}"]
            /
            (
                df[f"VV_{i:02d}"]
                + EPSILON
            )
        )

        ratio_cols.append(col)

    engineered_df["VH_VV_ratio_mean"] = (
        engineered_df[ratio_cols].mean(axis=1)
    )

    engineered_df["VH_VV_ratio_std"] = (
        engineered_df[ratio_cols].std(axis=1)
    )

    # VH - VV Difference
   
    diff_cols = []

    for i in range(1, 13):

        col = f"VH_VV_diff_{i:02d}"

        engineered_df[col] = (
            df[f"VH_{i:02d}"]
            - df[f"VV_{i:02d}"]
        )

        diff_cols.append(col)

    engineered_df["VH_VV_diff_mean"] = (
        engineered_df[diff_cols].mean(axis=1)
    )

    engineered_df["VH_VV_diff_std"] = (
        engineered_df[diff_cols].std(axis=1)
    )

    # Coefficient of Variation
    
    for group in FEATURE_GROUPS.keys():

        engineered_df[f"{group}_cv"] = (
            engineered_df[f"{group}_std"]
            /
            (
                engineered_df[f"{group}_mean"].abs()
                + EPSILON
            )
        )

    return engineered_df