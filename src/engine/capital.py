"""Capital accumulation dynamics and absorbing bankruptcy handling."""

import ctypes
from typing import Optional
import numpy as np

from .binding import NATIVE_LIB, is_native_available

# Capital at or below this level counts as bankrupt. Without a tolerance,
# K0 - T_ban * Phi evaluates to ~1e-17 instead of 0 for "exactly lethal" bans.
BANKRUPTCY_TOL = 1e-9


def update_capital(
    capital: np.ndarray,
    profits: np.ndarray,
    fixed_cost: float,
    cap: Optional[float] = None,
) -> np.ndarray:
    """Updates capital reserves based on stage profits and fixed overhead costs.

    K_{t+1} = min(cap, max(0, K_t + pi_t - Phi)).
    Bankrupted firms (capital <= 0) remain at 0 (absorbing state).
    Banned firms (profits == 0) lose fixed_cost per period.
    """
    capital = np.ascontiguousarray(capital, dtype=np.float64)
    profits = np.ascontiguousarray(profits, dtype=np.float64)
    N = len(capital)
    out_capital = np.zeros(N, dtype=np.float64)

    if is_native_available():
        NATIVE_LIB.update_capital(
            capital.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            profits.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(N),
            ctypes.c_double(fixed_cost),
            out_capital.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    else:
        is_solvent = capital > 0.0
        updated = capital + profits - fixed_cost
        out_capital = np.where(is_solvent, np.maximum(0.0, updated), 0.0)

    out_capital = np.where(out_capital <= BANKRUPTCY_TOL, 0.0, out_capital)
    if cap is not None:
        out_capital = np.minimum(out_capital, cap)
    return out_capital


def get_bankrupt_mask(capital: np.ndarray) -> np.ndarray:
    """Returns boolean array indicating bankrupted firms (capital <= 0)."""
    return np.asarray(capital) <= 0.0
