#!/usr/bin/env python3
"""
PokemonRL - Train an RL agent to play Pokemon Red.

Main entry point. Select which approach to train or run:
  - Baseline: KNN frame-based exploration with CnnPolicy (PPO)
  - V2: Coordinate-based exploration with MultiInputPolicy (PPO, recommended)
"""

import subprocess
import sys
import os

# Paths to the two training approach directories
BASELINE_DIR = os.path.join(os.path.dirname(__file__), "baseline")
V2_DIR = os.path.join(os.path.dirname(__file__), "v2")


def print_banner():
    """Display the application title banner."""
    print()
    print("=" * 50)
    print("     PokemonRL - Pokemon Red RL Agent")
    print("=" * 50)
    print()


def print_menu():
    """Display the interactive menu options."""
    print("Select an option:")
    print()
    print("  [1] Train Baseline  (KNN exploration, CnnPolicy)")
    print("  [2] Run Baseline    (play with trained Baseline model)")
    print()
    print("  [3] Train V2        (coordinate exploration, MultiInputPolicy)")
    print("  [4] Run V2          (play with trained V2 model)")
    print()
    print("  [q] Quit")
    print()


def check_rom():
    """Verify that the Pokemon Red ROM file exists in ROM_INPUT/."""
    rom_path = os.path.join(os.path.dirname(__file__), "ROM_INPUT", "PokemonRed.gb")
    if not os.path.exists(rom_path):
        print(f"[!] ROM not found at: {rom_path}")
        print("    Place your Pokemon Red ROM file (PokemonRed.gb) in the ROM_INPUT/ folder.")
        print()
        return False
    return True


def run_script(working_dir, script_name):
    """Launch a training/inference script as a subprocess."""
    script_path = os.path.join(working_dir, script_name)
    if not os.path.exists(script_path):
        print(f"[!] Script not found: {script_path}")
        return
    print(f"\n--- Running {script_name} from {os.path.basename(working_dir)}/ ---\n")
    subprocess.run([sys.executable, script_name], cwd=working_dir)


def main():
    """Main loop: display menu, dispatch to train/run scripts."""
    print_banner()

    while True:
        print_menu()
        choice = input(">>> ").strip().lower()

        if choice == "q":
            print("Bye!")
            break
        elif choice == "1":
            if not check_rom():
                continue
            run_script(BASELINE_DIR, "run_baseline_parallel_fast.py")
        elif choice == "2":
            if not check_rom():
                continue
            run_script(BASELINE_DIR, "run_pretrained_interactive.py")
        elif choice == "3":
            if not check_rom():
                continue
            run_script(V2_DIR, "baseline_fast_v2.py")
        elif choice == "4":
            if not check_rom():
                continue
            run_script(V2_DIR, "run_pretrained_interactive.py")
        else:
            print("Invalid choice. Try again.\n")


if __name__ == "__main__":
    main()
