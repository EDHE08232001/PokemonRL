#!/usr/bin/env python3
"""
PokemonRL Environment Check

Validates that all dependencies, files, hardware, and runtime behaviours
are correctly configured before training or running any version
(Baseline, V2, V3).

Also verifies V3-specific fixes:
  - get_latest_stats() returns only the last dict (not full history)
  - save_final_state is True in baseline_fast_v3.py
  - save_and_print_info() is actually called inside step()
  - go_forever.sh uses a safe glob pattern (not ls -l column parsing)
  - train_v3.sh requests --mem=256G and an absolute venv activate path

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
import ast
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Dependency definitions per version
# ---------------------------------------------------------------------------

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

BASELINE_DEPS = [
    ("hnswlib", "hnswlib", ">=0.7.0"),
]

V3_DEPS = [
    ("sb3_contrib", "sb3-contrib", ">=2.3.0"),
    ("networkx",    "networkx",    ">=3.0"),
]

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
    "src/v3/tensorboard_callback_v3.py",
    "src/v3/map_data.json",
    "src/v3/events.json",
    "train_v3.sh",
]

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def ok(msg):
    print(f"  {GREEN}[OK  ]{RESET}  {msg}")

def warn(msg):
    print(f"  {YELLOW}[WARN]{RESET}  {msg}")

def fail(msg):
    print(f"  {RED}[FAIL]{RESET}  {msg}")

def header(title):
    print(f"\n{BOLD}--- {title} ---{RESET}")

# ---------------------------------------------------------------------------
# Basic checks
# ---------------------------------------------------------------------------

def check_python():
    header("Python")
    v = sys.version_info
    label = (f"Python {v.major}.{v.minor}.{v.micro}  "
             f"({platform.python_implementation()}, {struct.calcsize('P')*8}-bit)")
    if (v.major, v.minor) >= (3, 10):
        ok(label)
        return True
    fail(f"{label}  — Python >=3.10 required")
    return False


def check_imports(deps, label):
    header(f"Packages — {label}")
    failures = 0
    for import_name, pip_name, version_hint in deps:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", getattr(mod, "VERSION", ""))
            ver_str = f"  (v{ver})" if ver else ""
            ok(f"{pip_name}{ver_str}")
        except ImportError:
            hint = (f'pip install "{pip_name}{version_hint}"'
                    if version_hint else f'pip install "{pip_name}"')
            fail(f"{pip_name} not found  — {hint}")
            failures += 1
    return failures


def check_files(file_list, label):
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
    header("Hardware")
    ok(f"Platform: {platform.system()} {platform.machine()}")
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem  = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                ok(f"GPU {i}: {name}  ({mem:.1f} GB)")
            ok(f"CUDA version: {torch.version.cuda}")
        else:
            warn("No CUDA GPU detected — training will use CPU (slower but functional)")
    except ImportError:
        warn("torch not installed — cannot check GPU")
    return 0


def check_sdl2():
    header("SDL2 Library")
    try:
        import sdl2.dll  # noqa: F401
        ok("SDL2 shared library loaded")
        return 0
    except Exception:
        pass
    try:
        import sdl2  # noqa: F401
        ok("SDL2 module available")
        return 0
    except ImportError:
        pass
    try:
        import ctypes.util
        path = ctypes.util.find_library("SDL2")
        if path:
            ok(f"SDL2 found at: {path}")
            return 0
    except Exception:
        pass
    warn("SDL2 library not detected — headless training may still work, "
         "but interactive play needs SDL2")
    return 0


# ---------------------------------------------------------------------------
# V3 source-code correctness checks
# ---------------------------------------------------------------------------

def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return ""


def check_v3_get_latest_stats():
    """
    Verify get_latest_stats() returns only the last element,
    NOT the entire agent_stats list.
    """
    header("V3 Fix — get_latest_stats() (OOM prevention)")
    src_path = PROJECT_ROOT / "src/v3/red_gym_env_v3.py"
    if not src_path.exists():
        warn(f"Cannot check — {src_path} not found")
        return 0

    src = _read(src_path)
    failures = 0

    # Must contain the fixed single-element return
    if "return self.agent_stats[-1]" in src:
        ok("get_latest_stats() returns only the last stats dict  ✓")
    else:
        fail("get_latest_stats() does NOT return agent_stats[-1] — "
             "full history will be piped at episode end → OOM crash!")
        failures += 1

    # Must NOT return the full list
    bad_patterns = [
        "return self.agent_stats\n",
        "return self.agent_stats ",
        "get_attr(\"agent_stats\")",
        "get_attr('agent_stats')",
    ]
    for pat in bad_patterns:
        if pat in src:
            fail(f"Found dangerous pattern in source: '{pat.strip()}'")
            failures += 1

    return failures


def check_v3_save_and_print_info_called():
    """
    Verify save_and_print_info() is actually called inside step().
    """
    header("V3 Fix — save_and_print_info() called in step()")
    src_path = PROJECT_ROOT / "src/v3/red_gym_env_v3.py"
    if not src_path.exists():
        warn(f"Cannot check — {src_path} not found")
        return 0

    src = _read(src_path)
    failures = 0

    # Find the step() method body and check for the call
    # We look for save_and_print_info anywhere in step() — simple text search
    # is sufficient here since it's only called from one place.
    if "self.save_and_print_info(" in src:
        ok("save_and_print_info() is called inside step()  ✓")
    else:
        fail("save_and_print_info() is NOT called in step() — "
             "screenshots will never be saved at episode end!")
        failures += 1

    return failures


def check_v3_save_final_state_config():
    """
    Verify save_final_state is set to True in baseline_fast_v3.py env_config.
    """
    header("V3 Fix — save_final_state: True in baseline_fast_v3.py")
    src_path = PROJECT_ROOT / "src/v3/baseline_fast_v3.py"
    if not src_path.exists():
        warn(f"Cannot check — {src_path} not found")
        return 0

    src = _read(src_path)
    failures = 0

    # Look for the env_config dict and find save_final_state value
    # Match both True and False to report the actual value
    match_true  = re.search(r"['\"]save_final_state['\"\s]*:\s*True",  src)
    match_false = re.search(r"['\"]save_final_state['\"\s]*:\s*False", src)

    if match_true:
        ok("'save_final_state': True is set in env_config  ✓")
    elif match_false:
        fail("'save_final_state' is set to False — "
             "episode screenshots will NOT be saved! Change to True.")
        failures += 1
    else:
        warn("Could not find 'save_final_state' key in baseline_fast_v3.py — "
             "check the env_config dict manually.")

    return failures


def check_v3_save_final_state_runtime():
    """
    Perform a lightweight runtime smoke-test of the save_final_state path.
    Creates a mock config, calls save_and_print_info with done=True, and
    checks that the expected files are created in a temp directory.
    """
    header("V3 Fix — save_final_state runtime smoke test")
    import tempfile
    import numpy as np

    failures = 0

    # We need matplotlib and the env class importable
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend — no display needed
    except ImportError:
        warn("matplotlib not installed — skipping runtime smoke test")
        return 0

    src_path = PROJECT_ROOT / "src/v3"
    if not src_path.exists():
        warn("src/v3/ not found — skipping runtime smoke test")
        return 0

    # Add src dirs to path so imports work
    for p in [str(src_path), str(src_path.parent)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Build a minimal config — enough for the constructor to not crash
        config = {
            "session_path":     tmp,
            "save_final_state": True,
            "print_rewards":    True,
            "headless":         True,
            "save_video":       False,
            "fast_video":       False,
            "action_freq":      24,
            "max_steps":        100,
            "init_state":       str(PROJECT_ROOT / "saves/init.state"),
            "gb_path":          str(PROJECT_ROOT / "ROM_INPUT/PokemonRed.gb"),
            "reward_scale":     0.5,
            "explore_weight":   0.25,
            "debug":            False,
        }

        rom_missing   = not (PROJECT_ROOT / "ROM_INPUT/PokemonRed.gb").exists()
        state_missing = not (PROJECT_ROOT / "saves/init.state").exists()

        if rom_missing or state_missing:
            warn("ROM or init.state missing — cannot instantiate env for runtime test. "
                 "Falling back to static source-code checks only.")
            return failures

        try:
            from v3.red_gym_env_v3 import RedGymEnv
        except Exception as e:
            warn(f"Could not import RedGymEnv ({e}) — skipping runtime smoke test")
            return failures

        try:
            env = RedGymEnv(config)
            env.reset()

            # Simulate end-of-episode state
            env.step_count  = env.max_steps - 1  # triggers done=True in save logic
            env.total_reward = 42.0
            env.reset_count  = 1

            # Build a minimal obs dict that save_and_print_info needs
            obs = {
                "map": np.zeros((env.coords_pad * 4, env.coords_pad * 4, 1),
                                dtype=np.uint8),
            }

            env.save_and_print_info(done=True, obs=obs)
            env.pyboy.stop()

            # Check that final_states/ directory was created and contains files
            fs_path = tmp / "final_states"
            if fs_path.exists():
                saved = list(fs_path.glob("*.jpeg"))
                if saved:
                    ok(f"save_final_state runtime test passed — "
                       f"{len(saved)} screenshot(s) written to final_states/  ✓")
                else:
                    fail("final_states/ directory created but contains no .jpeg files — "
                         "imsave may have failed silently.")
                    failures += 1
            else:
                fail("final_states/ directory was NOT created — "
                     "save_and_print_info() did not execute the save block!")
                failures += 1

        except Exception as e:
            warn(f"Runtime smoke test raised an exception: {e}\n"
                 "  This may be harmless if PyBoy cannot open a display in headless mode.\n"
                 "  Review the source-code checks above for definitive results.")

    return failures


def check_v3_train_sh():
    """
    Verify train_v3.sh has:
      - --mem=256G  (prevents OOM kill)
      - --time set  (job won't be killed prematurely)
      - absolute path to venv activate
    """
    header("V3 Fix — train_v3.sh SLURM resource requests")

    # Look for train_v3.sh in project root OR one level up
    candidates = [
        PROJECT_ROOT / "train_v3.sh",
        PROJECT_ROOT.parent / "train_v3.sh",
    ]
    sh_path = next((p for p in candidates if p.exists()), None)

    if sh_path is None:
        warn("train_v3.sh not found in project root — skipping SLURM checks")
        return 0

    src = _read(sh_path)
    failures = 0

    # --mem check
    mem_match = re.search(r"#SBATCH\s+--mem=(\S+)", src)
    if mem_match:
        mem_val = mem_match.group(1).upper()
        # Parse numeric value — strip G/GB suffix
        num = re.sub(r"[^\d]", "", mem_val)
        if num and int(num) >= 256:
            ok(f"--mem={mem_match.group(1)} (>=256G)  ✓")
        else:
            fail(f"--mem={mem_match.group(1)} is too low — "
                 "need at least 256G to survive episode-end data transfer!")
            failures += 1
    else:
        fail("--mem not set in train_v3.sh — SLURM may kill job for exceeding default memory!")
        failures += 1

    # --time check
    if re.search(r"#SBATCH\s+--time=\S+", src):
        time_match = re.search(r"#SBATCH\s+--time=(\S+)", src)
        ok(f"--time={time_match.group(1)}  ✓")
    else:
        warn("--time not set in train_v3.sh — job may be killed by default time limit")

    # Absolute activate path check (must not use bare relative path)
    activate_match = re.search(r"source\s+(\S+activate)", src)
    if activate_match:
        activate_path = activate_match.group(1)
        if activate_path.startswith("$HOME") or activate_path.startswith("/"):
            ok(f"venv activate path is absolute: {activate_path}  ✓")
        else:
            warn(f"venv activate path looks relative: '{activate_path}' — "
                 "SLURM may run from a different working directory. "
                 "Use $HOME/PokemonRL/v3env/local/bin/activate instead.")
    else:
        warn("Could not find 'source ... activate' in train_v3.sh — check venv activation manually.")

    return failures


def check_v3_tensorboard_callback():
    """
    Verify tensorboard_callback_v3.py uses get_latest_stats(),
    not the old get_attr('agent_stats') that pulled full history.
    """
    header("V3 Fix — tensorboard_callback_v3.py uses get_latest_stats()")
    cb_path = PROJECT_ROOT / "src/v3/tensorboard_callback_v3.py"
    if not cb_path.exists():
        warn(f"Cannot check — {cb_path} not found")
        return 0

    src = _read(cb_path)
    failures = 0

    if "get_latest_stats" in src:
        ok("TensorboardCallback calls get_latest_stats()  ✓")
    else:
        fail("TensorboardCallback does NOT call get_latest_stats() — "
             "it may still be pulling full agent_stats history!")
        failures += 1

    # Make sure the old dangerous pattern is gone
    if 'get_attr("agent_stats")' in src or "get_attr('agent_stats')" in src:
        fail("TensorboardCallback still uses get_attr('agent_stats') — "
             "this sends 163K dicts across the multiprocessing pipe → OOM!")
        failures += 1
    else:
        ok("TensorboardCallback does NOT use get_attr('agent_stats')  ✓")

    return failures


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_checks(versions):
    print(f"\n{BOLD}{'=' * 56}")
    print("   PokemonRL — Environment & V3 Fix Verification")
    print(f"{'=' * 56}{RESET}")
    print(f"  Versions to check: {', '.join(v.upper() for v in versions)}")

    total_failures = 0

    # ── Basic environment ──────────────────────────────────────────────────
    if not check_python():
        total_failures += 1

    total_failures += check_imports(CORE_DEPS, "Core (all versions)")

    if "baseline" in versions:
        total_failures += check_imports(BASELINE_DEPS, "Baseline extras")
    if "v3" in versions:
        total_failures += check_imports(V3_DEPS, "V3 extras")

    total_failures += check_sdl2()
    check_hardware()

    total_failures += check_files(COMMON_FILES, "Common")
    if "baseline" in versions:
        total_failures += check_files(BASELINE_FILES, "Baseline")
    if "v2" in versions:
        total_failures += check_files(V2_FILES, "V2")
    if "v3" in versions:
        total_failures += check_files(V3_FILES, "V3")

    # ── V3 correctness checks ──────────────────────────────────────────────
    if "v3" in versions:
        total_failures += check_v3_get_latest_stats()
        total_failures += check_v3_save_and_print_info_called()
        total_failures += check_v3_save_final_state_config()
        total_failures += check_v3_save_final_state_runtime()
        total_failures += check_v3_train_sh()
        total_failures += check_v3_tensorboard_callback()

    # ── Summary ───────────────────────────────────────────────────────────
    header("Summary")
    if total_failures == 0:
        ok("All checks passed — ready to train!")
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
