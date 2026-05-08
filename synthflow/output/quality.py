from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def distribution_similarity(real: pd.Series, synth: pd.Series) -> float:
    from scipy.stats import ks_2samp

    r = real.dropna().values.astype(float)
    s = synth.dropna().values.astype(float)
    if len(r) < 2 or len(s) < 2:
        return 0.0
    try:
        stat, _ = ks_2samp(r, s)
        return round(float(1.0 - stat), 4)
    except Exception:
        return 0.0


def distribution_similarity_all(real_df, synth_df, signal_cols):
    scores = {}
    for col in signal_cols:
        synth_col = f"synthetic_{col}"
        if col not in real_df.columns or synth_col not in synth_df.columns:
            continue
        scores[col] = distribution_similarity(real_df[col], synth_df[synth_col])
    if scores:
        scores["overall"] = round(float(np.mean(list(scores.values()))), 4)
    else:
        scores["overall"] = 0.0
    return scores


def autocorrelation_score(real_df, synth_df, signal_cols, lags=10):
    col_scores = []
    for col in signal_cols:
        synth_col = f"synthetic_{col}"
        if col not in real_df.columns or synth_col not in synth_df.columns:
            continue
        real_vals = real_df[col].dropna().values.astype(float)
        synth_vals = synth_df[synth_col].dropna().values.astype(float)
        if len(real_vals) < lags + 2 or len(synth_vals) < lags + 2:
            continue
        try:
            real_acf = [
                float(pd.Series(real_vals).autocorr(lag=i)) for i in range(1, lags + 1)
            ]
            synth_acf = [
                float(pd.Series(synth_vals).autocorr(lag=i)) for i in range(1, lags + 1)
            ]
            real_acf = [0.0 if np.isnan(v) else v for v in real_acf]
            synth_acf = [0.0 if np.isnan(v) else v for v in synth_acf]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr = float(np.corrcoef(real_acf, synth_acf)[0, 1])
            if np.isnan(corr):
                continue
            col_scores.append(float(np.clip(corr, 0.0, 1.0)))
        except Exception:
            continue
    if not col_scores:
        return 0.0
    return round(float(np.mean(col_scores)), 4)


def pca_overlap_score(real_df, synth_df, signal_cols, n_components=2):
    from sklearn.decomposition import PCA

    real_cols_avail = [c for c in signal_cols if c in real_df.columns]
    synth_cols_avail = [
        f"synthetic_{c}"
        for c in real_cols_avail
        if f"synthetic_{c}" in synth_df.columns
    ]
    if not real_cols_avail or not synth_cols_avail:
        return 0.0
    real_mat = real_df[real_cols_avail].dropna().values.astype(float)
    synth_mat = synth_df[synth_cols_avail].dropna().values.astype(float)
    if len(real_mat) < 2 or len(synth_mat) < 2:
        return 0.0
    n_comp = min(n_components, real_mat.shape[1], len(real_mat) - 1)
    if n_comp < 1:
        return 0.0
    try:
        pca = PCA(n_components=n_comp)
        real_proj = pca.fit_transform(real_mat)
        synth_proj = pca.transform(synth_mat)
        overlaps = []
        for i in range(n_comp):
            lo, hi = real_proj[:, i].min(), real_proj[:, i].max()
            if hi == lo:
                overlaps.append(1.0)
                continue
            buf = (hi - lo) * 0.10
            in_range = (
                (synth_proj[:, i] >= lo - buf) & (synth_proj[:, i] <= hi + buf)
            ).mean()
            overlaps.append(float(in_range))
        return round(float(np.mean(overlaps)), 4)
    except Exception:
        return 0.0


def tstr_score(real_df, synth_df, signal_cols):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    real_cols_avail = [c for c in signal_cols if c in real_df.columns]
    synth_cols_avail = [
        f"synthetic_{c}"
        for c in real_cols_avail
        if f"synthetic_{c}" in synth_df.columns
    ]
    if not real_cols_avail or len(real_cols_avail) != len(synth_cols_avail):
        return 0.0
    real_mat = real_df[real_cols_avail].dropna().values.astype(float)
    synth_mat = synth_df[synth_cols_avail].dropna().values.astype(float)
    n = min(len(real_mat), len(synth_mat), 500)
    if n < 10:
        return 0.0
    try:
        X = np.vstack([real_mat[:n], synth_mat[:n]])
        y = np.array([1] * n + [0] * n)
        clf = RandomForestClassifier(
            n_estimators=20, max_depth=4, random_state=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        accuracy = float(np.mean(scores))
        return round(float(np.clip(1.0 - (accuracy - 0.5) * 2, 0.0, 1.0)), 4)
    except Exception:
        return 0.0


def compute_all_metrics(real_df, synth_df, signal_cols):
    return {
        "distribution_similarity": distribution_similarity_all(
            real_df, synth_df, signal_cols
        ),
        "autocorrelation_score": autocorrelation_score(real_df, synth_df, signal_cols),
        "pca_overlap_score": pca_overlap_score(real_df, synth_df, signal_cols),
        "train_synth_test_real": tstr_score(real_df, synth_df, signal_cols),
    }
