"""
stats_utils.py
--------------
Reusable statistical testing functions for the LoL capstone project.
Every function returns a clean result dict so notebooks stay readable.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, binomtest, mannwhitneyu, spearmanr
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import ALPHA


# ── Effect sizes ──────────────────────────────────────────────────────────────

def cohens_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for two proportions.
    |h| < 0.2 = small, 0.2-0.5 = medium, > 0.5 = large
    """
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def interpret_cohens_h(h: float) -> str:
    """Human-readable effect size label."""
    h = abs(h)
    if h < 0.2:   return "small"
    if h < 0.5:   return "medium"
    return "large"


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cohen's d for difference between two means."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
        / (n1 + n2 - 2)
    )
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0


# ── Proportion tests ──────────────────────────────────────────────────────────

def test_win_rate(wins: int, total: int, h0_rate: float = 0.5,
                  label: str = "") -> dict:
    """
    Binomial test: is this win rate significantly different from h0_rate?

    Returns dict with: label, wins, total, observed_rate, p_value,
                       significant, effect_size, effect_label, ci_low, ci_high
    """
    result = binomtest(wins, total, h0_rate, alternative="two-sided")
    observed = wins / total
    h = cohens_h(observed, h0_rate)

    # Wilson score confidence interval
    z = 1.96
    p_hat = observed
    n = total
    denominator = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denominator
    margin  = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denominator
    ci_low  = max(0, centre - margin)
    ci_high = min(1, centre + margin)

    return {
        "label":        label,
        "wins":         wins,
        "total":        total,
        "observed_pct": round(observed * 100, 2),
        "p_value":      round(result.pvalue, 6),
        "significant":  result.pvalue < ALPHA,
        "effect_size":  round(abs(h), 4),
        "effect_label": interpret_cohens_h(h),
        "ci_low_pct":   round(ci_low * 100, 2),
        "ci_high_pct":  round(ci_high * 100, 2),
    }


def test_objective_impact(df: pd.DataFrame, obj_col: str,
                          label: str = "") -> dict:
    """
    Chi-square test: is securing this objective associated with winning?

    Returns dict with: label, chi2, p_value, significant, cramers_v,
                       win_rate_secured, win_rate_not_secured, odds_ratio, ci
    """
    # Only include games where someone got the objective
    secured_mask = df[obj_col] != 0
    sub = df[secured_mask].copy()
    sub["team_secured_won"] = (sub[obj_col] == sub["winner"]).astype(int)

    secured_won   = sub["team_secured_won"].sum()
    secured_lost  = len(sub) - secured_won
    # Games where nobody got it
    not_secured = df[~secured_mask]
    ns_t1_won = (not_secured["winner"] == 1).sum()
    ns_t2_won = (not_secured["winner"] == 2).sum()

    # Contingency table: [secured+won, secured+lost] vs [not_secured_t1_won, not_secured_t2_won]
    contingency = np.array([
        [secured_won, secured_lost],
        [ns_t1_won, ns_t2_won]
    ])

    chi2, p, dof, expected = chi2_contingency(contingency)

    # Cramer's V effect size
    n = contingency.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

    # Odds ratio
    a, b, c, d = secured_won, secured_lost, ns_t1_won, ns_t2_won
    odds_ratio = (a * d) / (b * c) if (b * c) > 0 else np.nan

    return {
        "label":               label,
        "chi2":                round(chi2, 4),
        "p_value":             round(p, 8),
        "significant":         p < ALPHA,
        "cramers_v":           round(cramers_v, 4),
        "effect_label":        "large" if cramers_v > 0.3 else "medium" if cramers_v > 0.1 else "small",
        "win_rate_secured_pct": round(secured_won / len(sub) * 100, 2),
        "n_secured":           len(sub),
        "n_not_secured":       len(not_secured),
        "odds_ratio":          round(odds_ratio, 3),
    }


# ── Non-parametric tests ──────────────────────────────────────────────────────

def test_duration_groups(group1: np.ndarray, group2: np.ndarray,
                         label1: str = "Group 1", label2: str = "Group 2") -> dict:
    """
    Mann-Whitney U test comparing two duration groups.
    Used when normality cannot be assumed (common for game duration distributions).
    """
    stat, p = mannwhitneyu(group1, group2, alternative="two-sided")
    d = cohens_d(group1, group2)

    return {
        "label1":     label1,
        "label2":     label2,
        "n1":         len(group1),
        "n2":         len(group2),
        "mean1":      round(np.mean(group1), 2),
        "mean2":      round(np.mean(group2), 2),
        "median1":    round(np.median(group1), 2),
        "median2":    round(np.median(group2), 2),
        "u_stat":     round(stat, 2),
        "p_value":    round(p, 6),
        "significant": p < ALPHA,
        "cohens_d":   round(d, 4),
        "effect_label": "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small",
    }


def test_normality(data: np.ndarray, label: str = "") -> dict:
    """
    Kolmogorov-Smirnov normality test.
    Returns whether data is normally distributed.
    """
    stat, p = stats.kstest(data, "norm",
                           args=(np.mean(data), np.std(data)))
    return {
        "label":       label,
        "ks_stat":     round(stat, 4),
        "p_value":     round(p, 6),
        "is_normal":   p > ALPHA,
        "n":           len(data),
        "mean":        round(np.mean(data), 2),
        "std":         round(np.std(data), 2),
        "skewness":    round(stats.skew(data), 4),
        "kurtosis":    round(stats.kurtosis(data), 4),
    }


def spearman_correlation(x: np.ndarray, y: np.ndarray,
                         label: str = "") -> dict:
    """Spearman rank correlation with interpretation."""
    corr, p = spearmanr(x, y)
    strength = (
        "very strong" if abs(corr) > 0.8 else
        "strong"      if abs(corr) > 0.6 else
        "moderate"    if abs(corr) > 0.4 else
        "weak"        if abs(corr) > 0.2 else
        "negligible"
    )
    direction = "positive" if corr > 0 else "negative"
    return {
        "label":       label,
        "correlation": round(corr, 4),
        "p_value":     round(p, 6),
        "significant": p < ALPHA,
        "strength":    strength,
        "direction":   direction,
    }


# ── Multiple testing correction ───────────────────────────────────────────────

def bonferroni_correction(p_values: list, alpha: float = ALPHA) -> list:
    """
    Apply Bonferroni correction to a list of p-values.
    Returns list of booleans: True = significant after correction.
    """
    if not p_values:
        return []
    n = len(p_values)
    corrected_alpha = alpha / n
    return [p < corrected_alpha for p in p_values]


def benjamini_hochberg(p_values: list, alpha: float = ALPHA) -> list:
    """
    Benjamini-Hochberg FDR correction — less conservative than Bonferroni.
    Better when testing many champions simultaneously.
    Returns list of booleans: True = significant after correction.
    """
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p   = np.array(p_values)[sorted_idx]
    thresholds = [(i + 1) / n * alpha for i in range(n)]
    significant = np.zeros(n, dtype=bool)

    # Find largest k where p(k) <= k/n * alpha
    for i in range(n - 1, -1, -1):
        if sorted_p[i] <= thresholds[i]:
            significant[sorted_idx[:i+1]] = True
            break

    return significant.tolist()


# ── Odds ratio with confidence interval ──────────────────────────────────────

def odds_ratio_ci(a: int, b: int, c: int, d: int,
                  confidence: float = 0.95) -> dict:
    """
    Odds ratio and confidence interval from a 2x2 contingency table.
    a=exposed+outcome, b=exposed+no outcome,
    c=unexposed+outcome, d=unexposed+no outcome
    """
    or_val = (a * d) / (b * c) if (b * c) > 0 else np.nan
    log_or = np.log(or_val) if or_val and or_val > 0 else np.nan
    se     = np.sqrt(1/a + 1/b + 1/c + 1/d) if min(a,b,c,d) > 0 else np.nan
    z      = stats.norm.ppf(1 - (1 - confidence) / 2)

    ci_low  = np.exp(log_or - z * se) if se else np.nan
    ci_high = np.exp(log_or + z * se) if se else np.nan

    return {
        "odds_ratio": round(or_val, 3) if or_val else None,
        "ci_low":     round(ci_low, 3) if ci_low else None,
        "ci_high":    round(ci_high, 3) if ci_high else None,
        "log_or":     round(log_or, 4) if log_or else None,
    }
