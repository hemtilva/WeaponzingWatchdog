"""C-types binding loader for the native C++ engine with NumPy fallback."""

import ctypes
import os
import platform
from pathlib import Path
from typing import Optional
import numpy as np


def _load_native_library() -> Optional[ctypes.CDLL]:
    """Attempts to locate and load the compiled engine shared library."""
    root_dir = Path(__file__).resolve().parent.parent
    csrc_dir = root_dir / "csrc"

    system = platform.system()
    if system == "Windows":
        lib_name = "engine.dll"
    elif system == "Darwin":
        lib_name = "engine.dylib"
    else:
        lib_name = "engine.so"

    lib_path = csrc_dir / lib_name
    if not lib_path.is_file():
        return None

    try:
        # On Windows, add potential compiler runtime DLL directories
        if system == "Windows" and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(csrc_dir))
            # OpenMP runtime (libgomp-1.dll / libomp.dll) from the MSYS2 toolchain
            msys_root = Path(os.environ.get("MSYS2_ROOT", r"C:\msys64"))
            for env_name in ("clangarm64", "ucrt64", "mingw64", "clang64"):
                msys_bin = msys_root / env_name / "bin"
                if msys_bin.is_dir():
                    os.add_dll_directory(str(msys_bin))

        dll = ctypes.CDLL(str(lib_path))

        # Setup ctypes function signatures
        # 1. compute_logit_shares
        dll.compute_logit_shares.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        dll.compute_logit_shares.restype = None

        # 2. compute_consumer_surplus
        dll.compute_consumer_surplus.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        dll.compute_consumer_surplus.restype = ctypes.c_double

        # 3. update_capital
        dll.update_capital.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_double),
        ]
        dll.update_capital.restype = None

        # 4. evaluate_rules
        dll.evaluate_rules.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_int),
        ]
        dll.evaluate_rules.restype = None

        # 5. update_ban_timers
        dll.update_ban_timers.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        dll.update_ban_timers.restype = None

        # 6. evaluate_counterfactual_harm
        dll.evaluate_counterfactual_harm.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_int),
        ]
        dll.evaluate_counterfactual_harm.restype = None

        # 7. solve_value_iteration_c
        dll.solve_value_iteration_c.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # price_grid
            ctypes.c_int,                     # num_prices
            ctypes.POINTER(ctypes.c_double),  # capital_grid
            ctypes.c_int,                     # num_caps
            ctypes.c_int,                     # T_ban
            ctypes.POINTER(ctypes.c_double),  # costs
            ctypes.POINTER(ctypes.c_double),  # qualities
            ctypes.c_double,                  # outside_option
            ctypes.c_double,                  # sigma_d
            ctypes.c_double,                  # market_size
            ctypes.c_double,                  # fixed_cost
            ctypes.c_int,                     # rule_type
            ctypes.c_double,                  # rule_param
            ctypes.POINTER(ctypes.c_double),  # nash_prices
            ctypes.c_double,                  # gamma
            ctypes.c_double,                  # tol
            ctypes.c_int,                     # max_iter
            ctypes.c_double,                  # mu_ethics
            ctypes.POINTER(ctypes.c_double),  # out_V
            ctypes.POINTER(ctypes.c_int),     # out_policy
        ]
        dll.solve_value_iteration_c.restype = None

        return dll
    except Exception as e:
        print(f"Warning: Failed to load native engine library: {e}")
        return None


NATIVE_LIB = _load_native_library()


def is_native_available() -> bool:
    """Returns True if the high-performance C++ backend is loaded."""
    return NATIVE_LIB is not None

