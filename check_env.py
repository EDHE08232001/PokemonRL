#!/usr/bin/env python3
"""
PokemonRL Environment Check

Validates that all dependencies, files, and hardware are properly configured
before training or running any version (Baseline, V2, V3).

Usage:
    python check_env.py              # Check all versions
    python check_env.py baseline     # Check Baseline only
    python check_env.py v2           # Check V2 only
    python check_env.py v3           # Check V3 only

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
"""

import importlib
import os
import platform
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Dependency definitions per version
# ---------------------------------------------------------------------------

# Core deps required by ALL versions
CORE_DEPS = [
    ("stable_baselines3", "stable-baselines3", ">=2.3.2"),
    ("gymnasium",         "gymnasium",          ">=0.29.1"),
    ("torch",             "torch",              ">=2.5.0"),
    ("pyboy",             "pyboy",              ">=2.4.0"),
    ("sdl2",              "PySDL2",             None),
    ("numpy",             "numpy",              ">=2.1.0"),
    ("einops",            "einops",             ">=0.8.0"),
    ("skimage",           "scikit-image",       ">=0.24.0"),
    ("scipy",             "scipy",              ">=1.14.0"),
    ("matplotlib",        "matplotlib",         ">=3.9.0"),
    ("pandas",            "pandas",             ">=2.2.0"),
    ("PIL",               "pillow",             ">=11.0.0"),
    ("mediapy",           "mediapy",            ">=1.2.2"),
    ("tensorboard",       "tensorboard",        ">=2.18.0"),
    ("websockets",        "websockets",         ">=13.1"),
]

# Extra deps per version
BASELINE_DEPS = [
    ("hnswlib", "hnswlib", ">=0.7.0"),
]

V3_DEPS = [
    ("sb3_contrib", "sb3-contrib", ">=2.3.0"),
    ("networkx",    "networkx",    ">=3.0"),
]

# Files required per version (relative to PROJECT_ROOT)
COMMON_FILES = [
    "ROM_INPUT/PokemonRed.gb",
]

BASELINE_FILES = [
    "saves/has_pokedex_nballs.state",
    "src/baseline/run_baseline_parallel_fast.py",
    "src/baseline/red_gym_env.py",
]

V2_FILES = [
    "saves/init.state",
    "src/v2/baseline_fast_v2.py",
    "src/v2/red_gym_env_v2.py",
]

V3_FILES = [
    "saves/init.state",
    "src/v3/baseline_fast_v3.py",
    "src/v3/red_gym_env_v3.py",
    "src/v3/map_data.json",
    "src/v3/events.json",
]

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ok(msg):
    print(f"  {GREEN}[OK]{RESET}  {msg}")


def warn(msg):
    print(f"  {YELLOW}[!!]{RESET}  {msg}")


def fail(msg):
    print(f"  {RED}[FAIL]{RESET} {msg}")


def header(title):
    print(f"\n{BOLD}--- {title} ---{RESET}")


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------

def check_python():
    """Verify Python version >= 3.10."""
    header("Python")
    v = sys.version_info
    label = f"Python {v.major}.{v.minor}.{v.micro}  ({platform.python_implementation()}, {struct.calcsize('P')*8}-bit)"
    if (v.major, v.minor) >= (3, 10):
        ok(label)
        return True
    else:
        fail(f"{label}  — Python >=3.10 required")
        return False


def check_imports(deps, label):
    """Try to import each dependency; return number of failures."""
    header(f"Packages — {label}")
    failures = 0
    for import_name, pip_name, version_hint in deps:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", getattr(mod, "VERSION", ""))
            ver_str = f"  (v{ver})" if ver else ""
            ok(f"{pip_name}{ver_str}")
        except ImportError:
            hint = f'pip install "{pip_name}{version_hint}"' if version_hint else f'pip install "{pip_name}"'
            fail(f"{pip_name} not found  — {hint}")
            failures += 1
    return failures


def check_files(file_list, label):
    """Check that each file exists; return number missing."""
    header(f"Files — {label}")
    missing = 0
    for rel in file_list:
        p = PROJECT_ROOT / rel
        if p.exists():
            ok(rel)
        else:
            fail(f"{rel}  — not found")
            missing += 1
    return missing


def check_hardware():
    """Report GPU / CUDA availability (informational, not a failure)."""
    header("Hardware")

    # CPU
    ok(f"Platform: {platform.system()} {platform.machine()}")

    # CUDA / GPU via torch
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
                ok(f"GPU {i}: {name}  ({mem:.1f} GB)")
            ok(f"CUDA version: {torch.version.cuda}")
        else:
            warn("No CUDA GPU detected — training will use CPU (slower but functional)")
    except ImportError:
        warn("torch not installed — cannot check GPU")

    return 0  # informational only


def check_sdl2():
    """Verify SDL2 shared library can be loaded (needed for PyBoy display)."""
    header("SDL2 Library")
    try:
        import sdl2.dll  # noqa: F401
        ok("SDL2 shared library loaded")
        return 0
    except ImportError:
        try:
            import sdl2  # noqa: F401
            ok("SDL2 module available")
            return 0
        except ImportError:
            pass
    except Exception:
        # Some SDL2 setups raise OSError instead of ImportError
        try:
            import ctypes.util
            path = ctypes.util.find_library("SDL2")
            if path:
                ok(f"SDL2 found at: {path}")
                return 0
        except Exception:
            pass

    warn("SDL2 library not detected — headless training may still work, but interactive play needs SDL2")
    return 0  # warn, don't fail — headless training can work without display


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_checks(versions):
    """Run all checks for the requested versions. Return True if all pass."""
    print(f"\n{BOLD}{'=' * 52}")
    print("   PokemonRL — Environment Check")
    print(f"{'=' * 52}{RESET}")
    print(f"  Versions to check: {', '.join(v.upper() for v in versions)}")

    total_failures = 0

    # 1. Python version
    if not check_python():
        total_failures += 1

    # 2. Core packages (always)
    total_failures += check_imports(CORE_DEPS, "Core (all versions)")

    # 3. Version-specific packages
    if "baseline" in versions:
        total_failures += check_imports(BASELINE_DEPS, "Baseline extras")
    if "v3" in versions:
        total_failures += check_imports(V3_DEPS, "V3 extras")

    # 4. SDL2 library
    total_failures += check_sdl2()

    # 5. Hardware info
    check_hardware()

    # 6. Common files
    total_failures += check_files(COMMON_FILES, "Common")

    # 7. Version-specific files
    if "baseline" in versions:
        total_failures += check_files(BASELINE_FILES, "Baseline")
    if "v2" in versions:
        total_failures += check_files(V2_FILES, "V2")
    if "v3" in versions:
        total_failures += check_files(V3_FILES, "V3")

    # Summary
    header("Summary")
    if total_failures == 0:
        ok("All checks passed — ready to train / run!")
    else:
        fail(f"{total_failures} check(s) failed — see above for details")

    print()
    return total_failures == 0


def main():
    valid = {"baseline", "v2", "v3"}
    args = [a.lower() for a in sys.argv[1:] if a.lower() in valid]
    versions = args if args else sorted(valid)

    passed = run_checks(versions)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
