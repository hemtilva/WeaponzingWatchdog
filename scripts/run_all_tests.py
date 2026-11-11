#!/usr/bin/env python3
"""Unified test runner for all test modules."""

import importlib
import inspect
import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

# Fallback mock for pytest if not installed in the environment
try:
    import pytest
except ImportError:
    from unittest.mock import MagicMock
    sys.modules["pytest"] = MagicMock()

TEST_MODULES = [
    "tests.test_proposition_1",
    "tests.test_logit_engine",
    "tests.test_capital_dynamics",
    "tests.test_counterfactual_harm",
    "tests.test_benchmarks",
    "tests.test_exact_dp",
    "tests.test_q_agent",
    "tests.test_traits",
    "tests.test_trainer",
    "tests.test_evaluation",
    "tests.test_extensions",
    "tests.test_scripted_attacker",
]


def run_all():
    print("=======================================================")
    print("Executing Full Test Suite (Weaponizing the Watchdog)")
    print("=======================================================")

    total_passed = 0
    total_failed = 0
    start_time = time.time()

    for mod_name in TEST_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f"[FAIL] Could not import {mod_name}: {e}")
            total_failed += 1
            continue

        test_funcs = [
            (name, func)
            for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_") and getattr(func, "__module__", "") == mod.__name__
        ]

        print(f"\n--- Running {mod_name} ({len(test_funcs)} test functions) ---")
        for name, func in test_funcs:
            try:
                func()
                print(f"  [PASS] {name}")
                total_passed += 1
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                total_failed += 1

    elapsed = time.time() - start_time
    print("\n=======================================================")
    print(f"Test Summary: {total_passed} Passed | {total_failed} Failed in {elapsed:.2f}s")
    print("=======================================================\n")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
