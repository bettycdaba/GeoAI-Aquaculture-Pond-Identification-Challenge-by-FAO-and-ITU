"""
Shared cross-validation scaffolding, built once so every model we compare in
notebook 05 (CatBoost, LightGBM, XGBoost, ...) uses IDENTICAL folds and an
IDENTICAL scoring function. Without this, "model A beat model B" could just
mean "we got lucky with which rows landed in which fold" rather than a real
difference.

Fold assignment happens at the ORIGIN level (1,821 unique locations), not the
augmented-row level (18,210 rows) — every one of a location's masked variants
always lands in the same fold, by construction, which is what prevents the
grouping leakage discussed in notebook 03.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from src.config import RANDOM_SEED


def assign_origin_folds(train_features: pd.DataFrame, n_splits: int = 5,
                          seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Add a `fold` column to train_features (0..n_splits-1), assigned per
    ORIGIN (so all variants of a location share one fold) and stratified by
    label at the origin level (each origin's label is constant across its
    variants, so this is exact, not approximate).
    """
    unique_origins = (
        train_features[["origin_id", "label"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    unique_origins["fold"] = -1
    for fold_i, (_, val_idx) in enumerate(
        skf.split(unique_origins, unique_origins["label"])
    ):
        unique_origins.loc[val_idx, "fold"] = fold_i

    return train_features.merge(
        unique_origins[["origin_id", "fold"]], on="origin_id", how="left"
    )


def composite_score(y_true, y_prob) -> tuple[float, float, float]:
    """
    The actual leaderboard metric: 0.6 * F1(@ fixed 0.5 threshold) + 0.4 * ROC-AUC.
    Returns (f1, auc, composite) so callers can report all three.
    """
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    composite = 0.6 * f1 + 0.4 * auc
    return f1, auc, composite


def compute_adversarial_oof(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    feature_cols: list[str],
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    For every row in train_features, produce an honest out-of-fold P(is_test)
    using the SAME `fold` column already assigned by `assign_origin_folds` —
    so this lines up exactly with whichever model's CV we're comparing it to.

    For fold i: train an adversarial classifier on (all other folds' train
    rows + all of test), predict on fold i's held-out train rows. Test rows
    don't need their own held-out split here since we're only scoring train
    rows' probabilities, not evaluating test-side accuracy.

    Reused for two purposes: (1) diagnosing whether a model's errors
    concentrate in high-shift regions of feature space, (2) generating
    sample weights for the reweighting experiment planned for the
    "refine the winner" stage.
    """
    import lightgbm as lgb  # local import: only this function needs it

    assert "fold" in train_features.columns, "call assign_origin_folds first"
    oof = np.zeros(len(train_features))

    for fold_i in sorted(train_features["fold"].unique()):
        tr_mask = (train_features["fold"] != fold_i).values
        val_mask = (train_features["fold"] == fold_i).values

        X_adv_train = pd.concat(
            [train_features.loc[tr_mask, feature_cols], test_features[feature_cols]],
            ignore_index=True,
        )
        y_adv_train = np.concatenate(
            [np.zeros(tr_mask.sum()), np.ones(len(test_features))]
        )

        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            is_unbalance=True, random_state=seed, verbosity=-1,
        )
        model.fit(X_adv_train, y_adv_train)
        oof[val_mask] = model.predict_proba(
            train_features.loc[val_mask, feature_cols]
        )[:, 1]

    return oof
