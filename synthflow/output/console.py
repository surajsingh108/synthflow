from __future__ import annotations


def bar(score: float, width: int = 14) -> str:
    filled = max(0, min(width, int(round(score * width))))
    return "█" * filled + "░" * (width - filled)


def print_summary(
    run_id,
    model,
    backend,
    n_samples,
    sampling_rate_hz,
    imputation_report,
    quality_metrics,
    signal_cols,
    synth_df,
):
    dist = quality_metrics.get("distribution_similarity", {})
    overall_dist = dist.get("overall", 0.0)
    autocorr = quality_metrics.get("autocorrelation_score", 0.0)
    pca = quality_metrics.get("pca_overlap_score", 0.0)
    tstr = quality_metrics.get("train_synth_test_real", 0.0)
    missing_before = imputation_report.get("missing_pct_before", 0.0)
    missing_after = imputation_report.get("missing_pct_after", 0.0)
    pattern = imputation_report.get("pattern_detected", "none")
    strategy = imputation_report.get("strategy_used", "none")
    synth_cols = [
        c
        for c in (synth_df.columns if synth_df is not None else [])
        if c.startswith("synthetic_")
    ]
    lines = [
        "=" * 56,
        " synthflow – Generation Report",
        "=" * 56,
        f"  Run ID     {run_id}",
        f"  Backend    {backend}  Model: {model}",
        f"  Samples    {n_samples} ({sampling_rate_hz} Hz)",
        "-" * 56,
        "  IMPUTATION",
        f"  Missing before    {missing_before:.1f}%",
        f"  Pattern           {pattern}",
        f"  Strategy          {strategy}",
        f"  Missing after     {missing_after:.1f}%  {'OK' if missing_after == 0 else 'WARN'}",
        "-" * 56,
        "  QUALITY METRICS",
        f"  Distribution   {bar(overall_dist)}  {overall_dist * 100:.0f}%",
        f"  Autocorrelation{bar(autocorr)}  {autocorr * 100:.0f}%",
        f"  PCA overlap    {bar(pca)}  {pca * 100:.0f}%",
        f"  TSTR score     {bar(tstr)}  {tstr * 100:.0f}%",
        "-" * 56,
        "  OUTPUT COLUMNS",
    ]
    for col in synth_cols[:6]:
        lines.append(f"  {col}")
    lines.append("=" * 56)
    output = "\n".join(lines)
    print(output)
    return output
