"""ctypes bridge to the native simulation loop (src/csrc/sim.cpp: run_simulation_c)."""

import ctypes
from typing import Dict, Any
import numpy as np

from .binding import NATIVE_LIB

_c_double_p = ctypes.POINTER(ctypes.c_double)
_c_int32_p = ctypes.POINTER(ctypes.c_int32)
_c_int64_p = ctypes.POINTER(ctypes.c_int64)
_c_float_p = ctypes.POINTER(ctypes.c_float)


class NativeSimParams(ctypes.Structure):
    # Field order and types must match `struct SimParams` in src/csrc/sim.cpp.
    _fields_ = [
        ("N", ctypes.c_int32), ("M", ctypes.c_int32),
        ("rule_type", ctypes.c_int32), ("penalty_mode", ctypes.c_int32),
        ("total_steps", ctypes.c_int32), ("warmup_steps", ctypes.c_int32),
        ("episode_length", ctypes.c_int32), ("eval_window", ctypes.c_int32),
        ("audit_budget", ctypes.c_int32), ("T_ban", ctypes.c_int32),
        ("freeze", ctypes.c_int32), ("has_cap", ctypes.c_int32),
        ("outside", ctypes.c_double), ("sigma", ctypes.c_double), ("Q", ctypes.c_double),
        ("fixed_cost", ctypes.c_double), ("K0", ctypes.c_double), ("cap", ctypes.c_double),
        ("rule_param", ctypes.c_double), ("ema_beta", ctypes.c_double),
        ("lr", ctypes.c_double), ("gamma", ctypes.c_double), ("eps_min", ctypes.c_double),
        ("eps_decay", ctypes.c_double), ("reward_eps", ctypes.c_double),
        ("seed", ctypes.c_uint64),
        ("price_grid", _c_double_p), ("costs", _c_double_p), ("qualities", _c_double_p),
        ("nash_prices", _c_double_p), ("nash_profits", _c_double_p),
        ("nash_idx", _c_int32_p), ("mono_idx", _c_int32_p), ("policies", _c_int32_p),
        ("alpha", _c_double_p), ("lambda_", _c_double_p), ("mu", _c_double_p),
        ("qtab", _c_float_p), ("visits", _c_int32_p), ("epsilon", _c_double_p),
    ]


class NativeSimOutput(ctypes.Structure):
    _fields_ = [
        ("mean_prices", _c_double_p), ("mean_profits", _c_double_p),
        ("induced_by", _c_int64_p), ("kills_by", _c_int64_p),
        ("counts", _c_int64_p), ("mean_cs", _c_double_p),
    ]


def is_native_sim_available() -> bool:
    return NATIVE_LIB is not None and hasattr(NATIVE_LIB, "run_simulation_c")


if is_native_sim_available():
    NATIVE_LIB.run_simulation_c.argtypes = [ctypes.POINTER(NativeSimParams), ctypes.POINTER(NativeSimOutput)]
    NATIVE_LIB.run_simulation_c.restype = ctypes.c_int32
    NATIVE_LIB.sim_num_states.argtypes = [ctypes.c_int32, ctypes.c_int32]
    NATIVE_LIB.sim_num_states.restype = ctypes.c_int64


def _ptr(arr: np.ndarray, ctype):
    return arr.ctypes.data_as(ctypes.POINTER(ctype))


def run_native(arrays: Dict[str, np.ndarray], scalars: Dict[str, Any]) -> Dict[str, Any]:
    """Runs the native loop. `arrays` must hold C-contiguous arrays of the right dtypes;
    qtab / visits / epsilon are updated in place."""
    N = int(scalars["N"])
    keep = {}  # keep converted arrays alive for the duration of the call

    def d(name):
        keep[name] = np.ascontiguousarray(arrays[name], dtype=np.float64)
        return _ptr(keep[name], ctypes.c_double)

    def i32(name):
        keep[name] = np.ascontiguousarray(arrays[name], dtype=np.int32)
        return _ptr(keep[name], ctypes.c_int32)

    qtab, visits, eps = arrays["qtab"], arrays["visits"], arrays["epsilon"]
    assert qtab.dtype == np.float32 and qtab.flags.c_contiguous
    assert visits.dtype == np.int32 and visits.flags.c_contiguous
    assert eps.dtype == np.float64 and eps.flags.c_contiguous

    p = NativeSimParams(
        **{k: scalars[k] for k in (
            "N", "M", "rule_type", "penalty_mode", "total_steps", "warmup_steps", "episode_length",
            "eval_window", "audit_budget", "T_ban", "freeze", "has_cap", "outside", "sigma", "Q",
            "fixed_cost", "K0", "cap", "rule_param", "ema_beta", "lr", "gamma", "eps_min",
            "eps_decay", "reward_eps", "seed",
        )},
        price_grid=d("price_grid"), costs=d("costs"), qualities=d("qualities"),
        nash_prices=d("nash_prices"), nash_profits=d("nash_profits"),
        nash_idx=i32("nash_idx"), mono_idx=i32("mono_idx"), policies=i32("policies"),
        alpha=d("alpha"), lambda_=d("lambda_"), mu=d("mu"),
        qtab=_ptr(qtab, ctypes.c_float), visits=_ptr(visits, ctypes.c_int32), epsilon=_ptr(eps, ctypes.c_double),
    )

    mean_prices = np.zeros(N, dtype=np.float64)
    mean_profits = np.zeros(N, dtype=np.float64)
    induced_by = np.zeros(N, dtype=np.int64)
    kills_by = np.zeros(N, dtype=np.int64)
    counts = np.zeros(8, dtype=np.int64)
    mean_cs = np.zeros(1, dtype=np.float64)
    o = NativeSimOutput(
        mean_prices=_ptr(mean_prices, ctypes.c_double), mean_profits=_ptr(mean_profits, ctypes.c_double),
        induced_by=_ptr(induced_by, ctypes.c_int64), kills_by=_ptr(kills_by, ctypes.c_int64),
        counts=_ptr(counts, ctypes.c_int64), mean_cs=_ptr(mean_cs, ctypes.c_double),
    )
    rc = NATIVE_LIB.run_simulation_c(ctypes.byref(p), ctypes.byref(o))
    if rc != 0:
        raise RuntimeError(f"run_simulation_c failed with code {rc}")

    keys = ("total_fresh_bans", "induced_bans", "total_sanctions", "induced_sanctions",
            "total_bankruptcies", "induced_bankruptcies", "episodes_in_eval", "eval_periods")
    res = {k: int(v) for k, v in zip(keys, counts)}
    res.update(mean_prices=mean_prices, mean_profits=mean_profits, induced_by=induced_by,
               kills_by=kills_by, mean_consumer_surplus=float(mean_cs[0]))
    return res


def num_states(N: int, M: int) -> int:
    return int(NATIVE_LIB.sim_num_states(N, M))
