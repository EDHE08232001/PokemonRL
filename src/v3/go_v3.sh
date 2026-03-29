SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$SCRIPT_DIR/runs"
NUM_EPISODES="${1:-1}" # defaults to 1 if no arggument is present

for i in $(seq 1 $NUM_EPISODES); do
    
done