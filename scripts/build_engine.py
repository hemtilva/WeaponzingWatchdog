#!/usr/bin/env python3
"""Cross-platform build script for native C++ simulation engine.

Auto-detects the host operating system, architecture, and available C++ compilers
(Clang++, MSVC, GCC) to compile src/csrc/engine.cpp into a shared library.
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def find_compiler():
    """Finds an appropriate C++ compiler on the system."""
    # Check for MSYS2 clangarm64 clang++ first (especially on Windows ARM64)
    msys_clang = Path(r"C:\msys64\clangarm64\bin\clang++.exe")
    if msys_clang.is_file():
        return str(msys_clang), "clang"

    # Search PATH for clang++, g++, or cl
    for compiler_name, compiler_type in [
        ("clang++", "clang"),
        ("g++", "gcc"),
        ("cl", "msvc"),
    ]:
        path = shutil.which(compiler_name)
        if path:
            return path, compiler_type

    return None, None


def build():
    root_dir = Path(__file__).resolve().parent.parent
    src_file = root_dir / "src" / "csrc" / "engine.cpp"
    out_dir = root_dir / "src" / "csrc"

    if not src_file.exists():
        print(f"Error: Source file not found at {src_file}", file=sys.stderr)
        sys.exit(1)

    system = platform.system()
    machine = platform.machine()
    print(f"Detected Platform: {system} ({machine})")

    compiler_path, compiler_type = find_compiler()
    if not compiler_path:
        print(
            "Error: No suitable C++ compiler (clang++, g++, cl) found in PATH or standard locations.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Using Compiler: {compiler_path} ({compiler_type})")

    # Determine shared library extension
    if system == "Windows":
        lib_name = "engine.dll"
    elif system == "Darwin":
        lib_name = "engine.dylib"
    else:
        lib_name = "engine.so"

    out_file = out_dir / lib_name

    # Build command flags
    if compiler_type in ("clang", "gcc"):
        cmd = [
            compiler_path,
            "-std=c++17",
            "-O3",
            "-shared",
        ]
        # Check OpenMP support
        if system == "Darwin":
            # On macOS, Apple Clang does not support -fopenmp directly
            brew_libomp = Path("/opt/homebrew/opt/libomp")
            intel_libomp = Path("/usr/local/opt/libomp")
            if (brew_libomp / "include" / "omp.h").is_file():
                cmd.extend([
                    "-Xpreprocessor", "-fopenmp",
                    f"-I{brew_libomp}/include",
                    f"-L{brew_libomp}/lib",
                    "-lomp",
                ])
            elif (intel_libomp / "include" / "omp.h").is_file():
                cmd.extend([
                    "-Xpreprocessor", "-fopenmp",
                    f"-I{intel_libomp}/include",
                    f"-L{intel_libomp}/lib",
                    "-lomp",
                ])
            else:
                print("Note: libomp not detected on macOS; compiling sequential native kernel.")
        else:
            cmd.append("-fopenmp")

        cmd.extend([
            str(src_file),
            "-o",
            str(out_file),
        ])
    elif compiler_type == "msvc":
        cmd = [
            compiler_path,
            "/std:c++17",
            "/O2",
            "/LD",
            "/openmp",
            str(src_file),
            f"/Fe:{out_file}",
        ]
    else:
        print(f"Unsupported compiler type: {compiler_type}", file=sys.stderr)
        sys.exit(1)

    print("Executing:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Compilation Failed!", file=sys.stderr)
        print("STDOUT:", result.stdout, file=sys.stderr)
        print("STDERR:", result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    print(f"Build succeeded! Binary created at: {out_file}")


if __name__ == "__main__":
    build()

