# Windows Setup Guide

> Originally contributed by Discord user **@Aisiktir**. Updated April 2026 to reflect the current project structure (`pyproject.toml`, Baseline/V2/V3 versions, `main.py` entry point).

---

## Prerequisites

### 1. Python 3.10+

Download and install the latest Python 3.10, 3.11, or 3.12 from [python.org](https://www.python.org/downloads/).

> **Important**: During installation, check **"Add Python to PATH"**.

To verify after installation, open Command Prompt (`cmd`) or PowerShell:

```cmd
python --version
```

You should see `Python 3.10.x` or higher.

### 2. Git

Download and install Git from [git-scm.com](https://git-scm.com/download/win).

Default installation options are fine. The key settings during install:

- **Adjusting your PATH**: Select "Git from the command line and also from 3rd-party software"
- **Line endings**: Select "Checkout as-is, commit Unix-style line endings"
- **Default editor**: Choose whichever editor you prefer (Notepad, VS Code, etc.)

Verify after installation:

```cmd
git --version
```

### 3. Microsoft C++ Build Tools

Some Python packages (e.g., `hnswlib` for Baseline) require compilation.

1. Download from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. In the installer, select **"Desktop development with C++"**
3. Make sure the following are checked:
   - **MSVC v143 (or later) - VS 2022 C++ x64/x86 build tools**
   - **Windows 10/11 SDK**

### 4. ffmpeg (Optional — for video recording)

If you want to record training videos:

1. Download from [ffmpeg.org](https://ffmpeg.org/download.html) (select a Windows build)
2. Extract and add the `bin/` folder to your system PATH

---

## Clone the Repository

```cmd
git clone https://github.com/EDHE08232001/PokemonRL.git
cd PokemonRL
```

---

## ROM Setup

You need a legally obtained **Pokemon Red** Game Boy ROM file.

1. Rename it to `PokemonRed.gb`
2. Place it in the `ROM_INPUT/` folder:

```
PokemonRL/
└── ROM_INPUT/
    └── PokemonRed.gb    ← place it here
```

3. (Optional) Verify the SHA1 checksum:

```cmd
certutil -hashfile ROM_INPUT\PokemonRed.gb SHA1
```

Expected: `ea9bcae617fdf159b045185467ae58b2e4a48b9a`

---

## Install Dependencies

Dependencies are managed via `pyproject.toml`. Choose the install command based on what you want to run:

```cmd
:: Core dependencies (V2 — recommended)
pip install .

:: Windows with NVIDIA GPU (includes CUDA 12.4 packages)
pip install ".[windows]"

:: V3 extras (RecurrentPPO with LSTM, topological graph)
pip install ".[v3]"

:: Baseline extras (KNN frame exploration via hnswlib)
pip install ".[baseline]"

:: Everything (all version extras)
pip install ".[all]"

:: Development tools (wandb, ipython, jupyter)
pip install ".[dev]"
```

> **Note**: On Windows with an NVIDIA GPU, `pip install ".[windows]"` will pull in the CUDA 12.4 runtime packages automatically. Make sure you have up-to-date [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) installed.

### Multiple Python Versions Installed?

If you have multiple Python versions, use the full path to the specific Python you want:

```cmd
"%localappdata%\Programs\Python\Python311\python.exe" -m pip install .
"%localappdata%\Programs\Python\Python311\python.exe" main.py
```

Or use `py` launcher to target a specific version:

```cmd
py -3.11 -m pip install .
py -3.11 main.py
```

---

## Verify Your Setup

Run the built-in environment checker to validate dependencies, ROM, hardware, and V3 source-code correctness:

```cmd
python check_env.py
```

Check a specific version only:

```cmd
python check_env.py v2          :: V2 only
python check_env.py v3          :: V3 only
python check_env.py baseline    :: Baseline only
```

This will report:

- Python version compatibility
- All required packages and their versions
- SDL2 library availability (needed for interactive play)
- GPU/CUDA detection
- ROM and save state file presence
- V3-specific source code correctness checks

---

## Running the Project

### Option A: Interactive Menu (Recommended)

```cmd
python main.py
```

This presents a menu:

```
  [1] Train Baseline  (KNN exploration, CnnPolicy)
  [2] Run Baseline    (play with trained Baseline model)

  [3] Train V2        (coordinate exploration, MultiInputPolicy)
  [4] Run V2          (play with trained V2 model)

  [5] Train V3        (LSTM + text + graph, RecurrentPPO)
  [6] Run V3          (play with trained V3 model)

  [c] Check Environment (verify deps, ROM, hardware)
  [q] Quit
```

### Option B: Run Training Scripts Directly

All scripts use **absolute path resolution** — they work from any working directory.

```cmd
:: Train V2 (recommended)
python src\v2\baseline_fast_v2.py

:: Train V3 (experimental — requires V3 extras)
python src\v3\baseline_fast_v3.py

:: Train Baseline
python src\baseline\run_baseline_parallel_fast.py
```

### Resume from Checkpoint

```cmd
:: V2
echo runs\poke_26214400_steps | python src\v2\baseline_fast_v2.py

:: V3
echo runs\poke_83886080_steps | python src\v3\baseline_fast_v3.py
```

### Run a Trained Model Interactively

```cmd
:: V2 — auto-detects latest checkpoint in src\v2\runs\
python src\v2\run_pretrained_interactive.py

:: V3 — auto-detects latest checkpoint in src\v3\runs\
python src\v3\run_pretrained_interactive.py

:: Baseline — edit file_name in the script to point to your checkpoint
python src\baseline\run_pretrained_interactive.py
```

The game renders in an SDL2 window at 6× speed. Toggle AI/manual control by creating `agent_enabled.txt` in the script's directory with `yes` or `no` on the first line.

---

## Performance Tips for Windows

- **Reduce `num_cpu`**: V2 and V3 default to 64 parallel environments. If your machine has fewer CPU cores (or limited RAM), lower this value in the training script (e.g., `num_cpu = 16` or `num_cpu = 8`).
- **GPU**: Training benefits from CUDA. Verify GPU detection with `python check_env.py` or:
  ```cmd
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
  ```
- **Antivirus**: Some antivirus software may slow down the emulators. Consider adding the project folder to your exclusion list.
- **Power plan**: Set Windows to "High Performance" power plan during training.

---

## Monitoring Training

### TensorBoard

```cmd
:: V2
tensorboard --logdir src\v2\runs\

:: V3
tensorboard --logdir src\v3\runs\

:: Baseline
tensorboard --logdir src\baseline\session_<id>\
```

Open [http://localhost:6006](http://localhost:6006) in your browser.

---

## Project Structure Overview

```
PokemonRL/
├── main.py                     # Entry point — interactive menu
├── check_env.py                # Environment & dependency validator
├── pyproject.toml              # Package config with optional dependency extras
├── ROM_INPUT/
│   └── PokemonRed.gb          # ← YOUR ROM FILE (user-supplied, gitignored)
├── saves/                      # Game Boy save states for training initialization
├── src/
│   ├── baseline/               # V1 — KNN frame exploration (CnnPolicy)
│   ├── v2/                     # V2 — Coordinate exploration (MultiInputPolicy)
│   └── v3/                     # V3 — LSTM + text + graph (RecurrentPPO)
├── train_v3.sh                 # SLURM submission script (HPC only)
├── pre_run_v3.sh               # Pre-training cleanup script (HPC only)
├── v3_setup_morning_star_hpc.md  # uOttawa Morning Star HPC setup guide
└── windows-setup-guide.md      # This file
```

See the [main README](README.md) for full documentation on each version, reward structures, hyperparameters, and architecture details.

---

## Troubleshooting

### `pip install .` fails with build errors

- Make sure Microsoft C++ Build Tools are installed (see [Prerequisites](#3-microsoft-c-build-tools))
- Try upgrading pip first: `python -m pip install --upgrade pip setuptools wheel`

### SDL2 errors when running interactively

- `pip install pysdl2-dll` should bundle SDL2. If it still fails, download SDL2 manually from [libsdl.org](https://www.libsdl.org/download-2.0.php) and add the DLL directory to your PATH.

### CUDA not detected

- Update your NVIDIA GPU drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx)
- Verify with: `python -c "import torch; print(torch.cuda.is_available())"`
- If using `pip install ".[windows]"`, CUDA runtime libraries are bundled — you do NOT need to install the full CUDA Toolkit separately.

### "Module not found" errors

- Make sure you installed the correct extras for the version you want to run:
  - V3 requires: `pip install ".[v3]"`
  - Baseline requires: `pip install ".[baseline]"`

### Multiple Python versions conflict

- Use `py -3.11` or the full path to the Python executable you want
- Consider using a virtual environment:
  ```cmd
  python -m venv venv
  venv\Scripts\activate
  pip install ".[windows]"
  ```
