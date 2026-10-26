"""Multinomial logit demand clearinghouse, profits, and consumer surplus."""

from dataclasses import dataclass, field
import ctypes
from typing import Optional, Sequence
import numpy as np

from .binding import NATIVE_LIB, is_native_available


@dataclass
class MarketConfig:
    """Market oligopoly configuration parameters."""
    num_firms: int = 3
    marginal_costs: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    qualities: np.ndarray = field(default_factory=lambda: np.full(3, 2.0, dtype=np.float64))
    outside_option: float = 0.0
    price_sensitivity: float = 0.25  # sigma_d > 0
    market_size: float = 1.0         # Q > 0
    fixed_cost: float = 0.1          # Phi > 0
    initial_capital: float = 1.0     # K_0
    ban_duration: int = 3            # T_ban
    audit_budget: int = 1            # B
    threshold_delta: float = 0.2     # delta


def compute_market_shares(
    prices: np.ndarray,
    qualities: np.ndarray,
    active_mask: np.ndarray,
    outside_option: float,
    sigma_d: float,
) -> np.ndarray:
    """Computes multinomial logit market shares for active firms with outside option.

    Inactive/banned firms receive share 0.0.
    """
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    qualities = np.ascontiguousarray(qualities, dtype=np.float64)
    active_mask = np.ascontiguousarray(active_mask, dtype=np.int32)
    N = len(prices)
    shares = np.zeros(N, dtype=np.float64)

    if is_native_available():
        NATIVE_LIB.compute_logit_shares(
            prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            qualities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            active_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int(N),
            ctypes.c_double(outside_option),
            ctypes.c_double(sigma_d),
            shares.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return shares

    # Vectorized NumPy Fallback
    active_idx = np.where(active_mask != 0)[0]
    if len(active_idx) == 0:
        return shares

    v_outside = outside_option / sigma_d
    v_active = (qualities[active_idx] - prices[active_idx]) / sigma_d
    max_v = max(v_outside, np.max(v_active))

    denom = np.exp(v_outside - max_v) + np.sum(np.exp(v_active - max_v))
    if denom > 1e-9:
        shares[active_idx] = np.exp(v_active - max_v) / denom

    return shares


def compute_consumer_surplus(
    prices: np.ndarray,
    qualities: np.ndarray,
    active_mask: np.ndarray,
    outside_option: float,
    sigma_d: float,
    Q: float,
) -> float:
    """Computes total market consumer surplus under logit demand."""
    prices = np.ascontiguousarray(prices, dtype=np.float64)
    qualities = np.ascontiguousarray(qualities, dtype=np.float64)
    active_mask = np.ascontiguousarray(active_mask, dtype=np.int32)
    N = len(prices)

    if is_native_available():
        return float(
            NATIVE_LIB.compute_consumer_surplus(
                prices.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                qualities.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                active_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ctypes.c_int(N),
                ctypes.c_double(outside_option),
                ctypes.c_double(sigma_d),
                ctypes.c_double(Q),
            )
        )

    # Vectorized NumPy Fallback
    active_idx = np.where(active_mask != 0)[0]
    v_outside = outside_option / sigma_d
    if len(active_idx) == 0:
        max_v = v_outside
        sum_exp = np.exp(v_outside - max_v)
    else:
        v_active = (qualities[active_idx] - prices[active_idx]) / sigma_d
        max_v = max(v_outside, np.max(v_active))
        sum_exp = np.exp(v_outside - max_v) + np.sum(np.exp(v_active - max_v))

    return float(Q * sigma_d * (max_v + np.log(sum_exp)))


def compute_profits(
    prices: np.ndarray,
    costs: np.ndarray,
    shares: np.ndarray,
    Q: float,
    active_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Computes firm stage operating profit: pi_i = Q * (p_i - c_i) * s_i."""
    profits = Q * (prices - costs) * shares
    if active_mask is not None:
        profits = np.where(active_mask != 0, profits, 0.0)
    return profits
