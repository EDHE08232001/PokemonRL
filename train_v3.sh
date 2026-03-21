#!/bin/bash
#SBATCH --account=small
#SBATCH --partition=small
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00

#SBATCH --output=console_outputs/v3_train-%j.out
#SBATCH --error=console_outputs/v3_train-%j.err

# Ensure the output directory exists
mkdir -p console_outputs

# Navigate to the project directory
cd $HOME/PokemonRL

# Activate your virtual environment
source $HOME/PokemonRL/v3env/local/bin/activate

# Run the V3 training script loop
bash src/v3/go_forever.sh
