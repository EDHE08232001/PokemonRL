# Morning Star V3 Environment Setup Guide

This guide ensures a clean, offline installation of the PokemonRL V3 pipeline on the Morning Star cluster. It is designed to bypass the compute node firewall and prevent the NumPy 2.0+ compatibility crash with system libraries.

## Step 1: Download the "Safe" Wheels (Local Computer)

On your personal computer (which has unrestricted internet access), create a folder and download all required packages. The explicit version limits (`"numpy<2.0.0"` and `"scikit-image>=0.24.0"`) force `pip` to resolve the exact compatible versions locally before uploading.

```bash
mkdir perfect_wheels
cd perfect_wheels

pip download stable-baselines3 gymnasium torch pyboy PySDL2 pysdl2-dll "numpy<2.0.0" einops "scikit-image>=0.24.0" scipy matplotlib pandas pillow mediapy tensorboard websockets sb3-contrib networkx --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.10
```

> **Note:** If your personal computer runs Linux, you can safely remove the `--platform manylinux2014_x86_64`, `--only-binary=:all:`, and `--python-version 3.10` flags.

Zip the `perfect_wheels` folder and upload `perfect_wheels.zip` to your `~/PokemonRL` directory via the Morning Star JupyterLab portal.

---

## Step 2: Create a Fresh Environment (Morning Star)

Open your Morning Star terminal and create a clean virtual environment from scratch.

```bash
cd ~/PokemonRL

# Create a clean environment
python3.10 -m venv v3env

# Activate it
source v3env/local/bin/activate
```

---

## Step 3: Extract and Install Offline

Unzip your uploaded wheels and run the offline install command. Because you pre-filtered the NumPy version in Step 1, this will install perfectly without clashing with the pre-installed system libraries.

```bash
# Unzip the bundle
unzip perfect_wheels.zip

# Install everything directly from the local folder
pip install --no-index --find-links ./perfect_wheels stable-baselines3 gymnasium torch pyboy PySDL2 pysdl2-dll "numpy<2.0.0" einops "scikit-image>=0.24.0" scipy matplotlib pandas pillow mediapy tensorboard websockets sb3-contrib networkx
```

---

## Step 4: Clean Up and Verify

Immediately delete the heavy wheel files to protect your 50GB storage quota, then run the checker.

```bash
# Delete the installation files
rm -rf perfect_wheels perfect_wheels.zip

# Verify the environment
python ./check_env.py v3
```

If the check returns **0 checks failed**, your environment is completely configured. You can now start training by running:

```bash
sbatch train_v3.sh
```
