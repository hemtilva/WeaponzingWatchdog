"""Statistical aggregation, non-parametric tests, and bootstrapping routines."""

from typing import Dict, Any, Tuple, Optional, Callable
import numpy as np
from scipy import stats


def bootstrap_ci(
    data: np.ndarray,
    num_bootstrap: int = 1000,
    ci: float = 0.95,
    statistic: Callable[[np.ndarray], float] = np.mean,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Computes empirical bootstrap confidence intervals for a metric.

    Returns:
        (ci_lower, ci_upper)
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0
    if len(arr) == 1:
        val = float(arr[0])
        return val, val

    if rng is None:
        rng = np.random.default_rng(42)

    boot_stats = np.empty(num_bootstrap, dtype=np.float64)
    n = len(arr)
    for b in range(num_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_stats[b] = statistic(sample)

    alpha_tail = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_stats, 100.0 * alpha_tail))
    upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha_tail)))
    return lower, upper


def compute_summary_stats(
    data: np.ndarray,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """Computes sample mean, std, median, IQR, and bootstrap confidence intervals."""
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "n": 0,
        }

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    median_val = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_val = float(q75 - q25)

    ci_low, ci_high = bootstrap_ci(arr, num_bootstrap=1000, ci=ci, rng=rng)

    return {
        "mean": mean_val,
        "std": std_val,
        "median": median_val,
        "iqr": iqr_val,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "n": len(arr),
    }


def test_hypothesis_h1(
    w_mean: np.ndarray,
    w_median: np.ndarray,
    w_cap: np.ndarray,
) -> Dict[str, Any]:
    """Tests Hypothesis H1 (Rule Vulnerability Ranking):

    H1 posits: W(Mean) > W(Median) >= W(Cap).
    Applies Mann-Whitney U tests to check for significant differences.
    """
    w_mean = np.asarray(w_mean, dtype=np.float64)
    w_median = np.asarray(w_median, dtype=np.float64)
    w_cap = np.asarray(w_cap, dtype=np.float64)

    # 1. Test Mean vs Median (one-sided: mean > median)
    stat_mm, p_val_mm = stats.mannwhitneyu(w_mean, w_median, alternative="greater")

    # 2. Test Mean vs Cap (one-sided: mean > cap)
    stat_mc, p_val_mc = stats.mannwhitneyu(w_mean, w_cap, alternative="greater")

    # 3. Test Median vs Cap (two-sided)
    stat_med_cap, p_val_med_cap = stats.mannwhitneyu(w_median, w_cap, alternative="two-sided")

    mean_mean = float(np.mean(w_mean))
    mean_median = float(np.mean(w_median))
    mean_cap = float(np.mean(w_cap))

    ranking_holds = (mean_mean > mean_median) and (mean_median >= mean_cap) and (p_val_mm < 0.05) and (p_val_mc < 0.05)

    return {
        "mean_w": mean_mean,
        "median_w": mean_median,
        "cap_w": mean_cap,
        "p_val_mean_gt_median": float(p_val_mm),
        "p_val_mean_gt_cap": float(p_val_mc),
        "p_val_median_vs_cap": float(p_val_med_cap),
        "hypothesis_supported": bool(ranking_holds),
    }


def test_hypothesis_h2(
    k0_over_phi: np.ndarray,
    t_ban: np.ndarray,
    w_rates: np.ndarray,
) -> Dict[str, Any]:
    """Tests Hypothesis H2 (Capital and Ban Duration Threshold):

    H2 posits that framing emerges predominantly when T_ban >= K0 / Phi
    (i.e., when suspension is long enough to force permanent bankruptcy).
    """
    k0_ratios = np.asarray(k0_over_phi, dtype=np.float64)
    tb = np.asarray(t_ban, dtype=np.float64)
    w = np.asarray(w_rates, dtype=np.float64)

    is_lethal_regime = tb >= k0_ratios
    w_lethal = w[is_lethal_regime]
    w_sublethal = w[~is_lethal_regime]

    mean_lethal = float(np.mean(w_lethal)) if len(w_lethal) > 0 else 0.0
    mean_sublethal = float(np.mean(w_sublethal)) if len(w_sublethal) > 0 else 0.0

    p_val = 1.0
    if len(w_lethal) > 0 and len(w_sublethal) > 0:
        _, p_val = stats.mannwhitneyu(w_lethal, w_sublethal, alternative="greater")

    supported = (mean_lethal > mean_sublethal) and (float(p_val) < 0.05)

    return {
        "mean_w_lethal": mean_lethal,
        "mean_w_sublethal": mean_sublethal,
        "difference": mean_lethal - mean_sublethal,
        "p_value": float(p_val),
        "hypothesis_supported": bool(supported),
    }


def test_hypothesis_h3(
    trait_values: np.ndarray,
    w_rates: np.ndarray,
    trait_name: str = "ethics_mu",
) -> Dict[str, Any]:
    """Tests Hypothesis H3 (Trait Modulation):

    H3 posits that weaponization rate W decreases monotonically with
    increasing ethics mu, altruism lambda, or risk aversion alpha.
    Evaluated using Spearman's rank correlation (rho < 0).
    """
    tv = np.asarray(trait_values, dtype=np.float64)
    w = np.asarray(w_rates, dtype=np.float64)

    corr, p_val = stats.spearmanr(tv, w)
    is_negative = corr < 0 and p_val < 0.05

    return {
        "trait": trait_name,
        "spearman_rho": float(corr),
        "p_value": float(p_val),
        "monotonic_decrease_supported": bool(is_negative),
    }


# These are library functions, not pytest tests; stop pytest from collecting them
# when they are imported into a test module.
test_hypothesis_h1.__test__ = False
test_hypothesis_h2.__test__ = False
test_hypothesis_h3.__test__ = False
