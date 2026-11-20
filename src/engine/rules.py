"""Regulatory rules, audit sampling, ban timer updates, and counterfactual harm."""

from enum import IntEnum
import ctypes
from typing import Optional, Union
import numpy as np

from .binding import NATIVE_LIB, is_native_available


class RuleType(IntEnum):
    """Static regulatory rule types."""
    MEAN_RATIO = 0
    MEDIAN_RATIO = 1
    PRICE_CAP = 2
    Z_SCORE = 3
    EMA_MEAN_RATIO = 4  # Defended rule: Exponential Moving Average smoothed mean
    PLANNER = 5         # Omniscient planner: flags only prices above the firm's own Nash price


def evaluate_rules(
    prices: np.ndarray,
    active_mask: np.ndarray,
    rule_type: Union[RuleType, int],
    param: float,
    ema_ref: Optional[float] = None,
    beta: float = 0.85,
    min_active: int = 3,
    nash_prices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Evaluates regulatory flags under the specified rule.

    Flags are evaluated only when N_active >= min_active (N_min, default 3). The PLANNER
    rule needs `nash_prices` (it knows costs, hence each firm's Nash price).
    """
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    active_mask = np.ascontiguousarray(active_mask, dtype=np.int32)
    N = len(prices)
    flags = np.zeros(N, dtype=np.int32)
    rule_type_val = int(rule_type)

    if is_native_available() and rule_type_val < 4 and min_active == 3:
        NATIVE_LIB.evaluate_rules(
            prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            active_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(N),
            ctypes.c_int(rule_type_val),
            ctypes.c_double(param),
            flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return flags

    # NumPy Fallback (and native implementation for EMA_MEAN_RATIO)
    active_indices = np.where(active_mask != 0)[0]
    n_act = len(active_indices)
    if n_act < min_active:
        return flags

    act_prices = prices[active_indices]

    if rule_type_val == RuleType.MEAN_RATIO:
        r_t = np.mean(act_prices)
        flagged = act_prices > (1.0 + param) * r_t
        flags[active_indices[flagged]] = 1
    elif rule_type_val == RuleType.EMA_MEAN_RATIO:
        # Defended moving average benchmark:
        # r_t = beta * r_{t-1} + (1 - beta) * mean(p_t)
        # Prevents overnight collapse from transient single-period undercutting crashes
        cur_mean = float(np.mean(act_prices))
        if ema_ref is not None and ema_ref > 0.0:
            r_t = float(beta * ema_ref + (1.0 - beta) * cur_mean)
        else:
            r_t = cur_mean
        flagged = act_prices > (1.0 + param) * r_t
        flags[active_indices[flagged]] = 1
    elif rule_type_val == RuleType.MEDIAN_RATIO:
        r_t = np.median(act_prices)
        flagged = act_prices > (1.0 + param) * r_t
        flags[active_indices[flagged]] = 1
    elif rule_type_val == RuleType.PRICE_CAP:
        flagged = act_prices > param
        flags[active_indices[flagged]] = 1
    elif rule_type_val == RuleType.Z_SCORE:
        std_p = np.std(act_prices)
        if std_p > 1e-7:
            z_scores = (act_prices - np.mean(act_prices)) / std_p
            flagged = z_scores > param
            flags[active_indices[flagged]] = 1
    elif rule_type_val == RuleType.PLANNER:
        if nash_prices is None:
            raise ValueError("PLANNER rule needs nash_prices")
        ref = np.asarray(nash_prices, dtype=np.float64)[active_indices]
        flags[active_indices[act_prices > ref + 1e-9]] = 1

    return flags


def rule_scores(
    prices: np.ndarray,
    active_mask: np.ndarray,
    rule_type: Union[RuleType, int],
    ema_ref: Optional[float] = None,
    beta: float = 0.85,
    nash_prices: Optional[np.ndarray] = None,
) -> np.ndarray:
    """The statistic each rule thresholds (higher = more suspicious), used by top-score-first
    audits: p / reference for the ratio rules, p for the cap, z for the z-score, p - p^N for
    the planner. Inactive firms score 0. Mirrors Sim::flags in src/csrc/sim.cpp."""
    prices = np.asarray(prices, dtype=np.float64)
    idx = np.where(np.asarray(active_mask) != 0)[0]
    out = np.zeros(len(prices), dtype=np.float64)
    if len(idx) == 0:
        return out
    ap = prices[idx]
    mean = float(np.mean(ap))
    rt = int(rule_type)
    if rt in (RuleType.MEAN_RATIO, RuleType.EMA_MEAN_RATIO, RuleType.MEDIAN_RATIO):
        if rt == RuleType.MEAN_RATIO:
            r = mean
        elif rt == RuleType.EMA_MEAN_RATIO:
            r = beta * ema_ref + (1.0 - beta) * mean if (ema_ref is not None and ema_ref > 0.0) else mean
        else:
            r = float(np.median(ap))
        out[idx] = ap / r if r > 0.0 else ap
    elif rt == RuleType.PRICE_CAP:
        out[idx] = ap
    elif rt == RuleType.Z_SCORE:
        sd = float(np.std(ap))
        if sd > 1e-7:
            out[idx] = (ap - mean) / sd
    elif rt == RuleType.PLANNER:
        out[idx] = ap - np.asarray(nash_prices, dtype=np.float64)[idx]
    return out


def select_top_audit_set(
    active_mask: np.ndarray,
    scores: np.ndarray,
    budget: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Top-score-first audit: the `budget` active firms with the highest rule score, equal
    scores in random order (random permutation, then a stable sort by score)."""
    idx = np.where(np.asarray(active_mask) != 0)[0]
    audited = np.zeros(len(active_mask), dtype=np.int32)
    if len(idx) == 0:
        return audited
    perm = idx[rng.permutation(len(idx))]
    order = perm[np.argsort(-np.asarray(scores)[perm], kind="stable")]
    audited[order[:min(budget, len(idx))]] = 1
    return audited


def sample_audit_set(
    active_mask: np.ndarray,
    budget: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Samples up to `budget` active firms uniformly at random for regulatory audit."""
    if rng is None:
        rng = np.random.default_rng()

    active_indices = np.where(active_mask != 0)[0]
    audited = np.zeros(len(active_mask), dtype=np.int32)
    if len(active_indices) == 0:
        return audited

    sample_size = min(budget, len(active_indices))
    chosen = rng.choice(active_indices, size=sample_size, replace=False)
    audited[chosen] = 1
    return audited


def update_ban_timers(
    timers: np.ndarray,
    flags: np.ndarray,
    audited: np.ndarray,
    T_ban: int,
) -> np.ndarray:
    """Updates ban countdown timers for all firms."""
    timers = np.ascontiguousarray(timers, dtype=np.int32)
    flags = np.ascontiguousarray(flags, dtype=np.int32)
    audited = np.ascontiguousarray(audited, dtype=np.int32)
    N = len(timers)
    out_timers = np.zeros(N, dtype=np.int32)

    if is_native_available():
        NATIVE_LIB.update_ban_timers(
            timers.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            flags.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            audited.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(N),
            ctypes.c_int(T_ban),
            out_timers.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return out_timers

    # NumPy Fallback
    is_penalized = (audited != 0) & (flags != 0)
    out_timers = np.where(is_penalized, T_ban, np.maximum(0, timers - 1))
    return out_timers


def compute_counterfactual_harm(
    prices: np.ndarray,
    nash_prices: np.ndarray,
    active_mask: np.ndarray,
    audited: np.ndarray,
    rule_type: Union[RuleType, int],
    param: float,
    ema_ref: Optional[float] = None,
    beta: float = 0.85,
    min_active: int = 3,
) -> np.ndarray:
    """Computes counterfactual harm attribution for all firms."""
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    nash_prices = np.ascontiguousarray(nash_prices, dtype=np.float64)
    active_mask = np.ascontiguousarray(active_mask, dtype=np.int32)
    audited = np.ascontiguousarray(audited, dtype=np.int32)
    N = len(prices)
    harm = np.zeros(N, dtype=np.int32)
    rule_type_val = int(rule_type)

    if is_native_available() and rule_type_val < 4 and min_active == 3:
        NATIVE_LIB.evaluate_counterfactual_harm(
            prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nash_prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            active_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            audited.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(N),
            ctypes.c_int(rule_type_val),
            ctypes.c_double(param),
            harm.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
        return harm

    # NumPy Fallback
    actual_flags = evaluate_rules(prices, active_mask, rule_type_val, param, ema_ref=ema_ref, beta=beta,
                                  min_active=min_active, nash_prices=nash_prices)
    for i in range(N):
        if active_mask[i] == 0:
            continue
        cf_prices = prices.copy()
        cf_prices[i] = nash_prices[i]
        cf_flags = evaluate_rules(cf_prices, active_mask, rule_type_val, param, ema_ref=ema_ref, beta=beta,
                                  min_active=min_active, nash_prices=nash_prices)

        for j in range(N):
            if j != i and audited[j] != 0:
                if actual_flags[j] == 1 and cf_flags[j] == 0:
                    harm[i] += 1

    return harm

