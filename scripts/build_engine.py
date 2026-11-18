#!/usr/bin/env python3
"""Cross-platform build script for the native C++ simulation engine.

Compiles every src/csrc/*.cpp into one shared library (engine.dll / .so / .dylib).
Auto-detects a compiler: MSYS2 toolchains on Windows, then clang++ / g++ / cl on PATH.
OpenMP is used when its runtime is installed; otherwise the build falls back to a
sequential library (OpenMP only parallelizes the exact-DP value iteration).

  python scripts/build_engine.py            # auto
  python scripts/build_engine.py --no-openmp
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def find_compiler():
    """Finds an appropriate C++ compiler on the system."""
    msys_root = Path(os.environ.get("MSYS2_ROOT", r"C:\msys64"))
    msys_candidates = [
        ("clangarm64", "clang++.exe", "clang"),  # Windows ARM64 (Snapdragon)
        ("ucrt64", "g++.exe", "gcc"),
        ("mingw64", "g++.exe", "gcc"),
        ("clang64", "clang++.exe", "clang"),
    ]
    if platform.machine().lower() not in ("arm64", "aarch64"):
        msys_candidates = msys_candidates[1:]
    if platform.system() == "Windows":
        for env, exe, ctype in msys_candidates:
            candidate = msys_root / env / "bin" / exe
            if candidate.is_file():
                return str(candidate), ctype

    for compiler_name, compiler_type in [("clang++", "clang"), ("g++", "gcc"), ("cl", "msvc")]:
        path = shutil.which(compiler_name)
        if path:
            return path, compiler_type
    return None, None


def openmp_flags(system: str):
    if system == "Darwin":
        for prefix in (Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp")):
            if (prefix / "include" / "omp.h").is_file():
                return ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include", f"-L{prefix}/lib", "-lomp"]
        return None
    return ["-fopenmp"]


def gnu_command(compiler_path, compiler_type, system, sources, out_file, use_openmp):
    cmd = [compiler_path, "-std=c++17", "-O3", "-shared"]
    if system != "Darwin":
        # Apple clang on Apple Silicon may reject -march=native; its default
        # target is already tuned for the host.
        cmd.append("-march=native")
    if use_openmp:
        cmd.extend(use_openmp)
    if system == "Windows" and compiler_type == "gcc":
        # Self-contained C++ runtime; libgomp (if used) is resolved at load time from
        # the MSYS2 bin dir by src/engine/binding.py.
        cmd.extend(["-static-libgcc", "-static-libstdc++"])
    if system != "Windows":
        cmd.append("-fPIC")
    cmd.extend([str(s) for s in sources])
    cmd.extend(["-o", str(out_file)])
    return cmd


def run(cmd, compiler_path):
    print("Executing:", " ".join(cmd))
    # The compiler's helper programs (cc1plus, ld) need their own bin dir on PATH,
    # otherwise MSYS2 g++ fails silently with no output.
    env = os.environ.copy()
    env["PATH"] = str(Path(compiler_path).parent) + os.pathsep + env.get("PATH", "")
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def build(no_openmp: bool = False):
    root_dir = Path(__file__).resolve().parent.parent
    csrc = root_dir / "src" / "csrc"
    sources = sorted(csrc.glob("*.cpp"))
    if not sources:
        print(f"Error: no C++ sources in {csrc}", file=sys.stderr)
        sys.exit(1)

    system = platform.system()
    print(f"Detected Platform: {system} ({platform.machine()})")
    compiler_path, compiler_type = find_compiler()
    if not compiler_path:
        print("Error: no C++ compiler (MSYS2, clang++, g++, cl) found.", file=sys.stderr)
        sys.exit(1)
    print(f"Using Compiler: {compiler_path} ({compiler_type})")

    lib_name = {"Windows": "engine.dll", "Darwin": "engine.dylib"}.get(system, "engine.so")
    out_file = csrc / lib_name

    if compiler_type == "msvc":
        cmd = [compiler_path, "/std:c++17", "/O2", "/LD"]
        if not no_openmp:
            cmd.append("/openmp")
        cmd.extend([str(s) for s in sources] + [f"/Fe:{out_file}"])
        result = run(cmd, compiler_path)
    else:
        omp = None if no_openmp else openmp_flags(system)
        result = run(gnu_command(compiler_path, compiler_type, system, sources, out_file, omp), compiler_path)
        if result.returncode != 0 and omp and ("gomp" in result.stderr or "omp" in result.stderr):
            print("OpenMP runtime not available; rebuilding without OpenMP (DP solver runs single-threaded).")
            result = run(gnu_command(compiler_path, compiler_type, system, sources, out_file, None), compiler_path)

    if result.returncode != 0:
        print("Compilation Failed!", file=sys.stderr)
        print("STDOUT:", result.stdout, file=sys.stderr)
        print("STDERR:", result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    print(f"Build succeeded! Binary created at: {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-openmp", action="store_true")
    build(no_openmp=ap.parse_args().no_openmp)
