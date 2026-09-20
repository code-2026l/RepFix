#!/bin/bash
#SBATCH --job-name=repfix_stf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=03:00:00
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.err

# Portability note: data root defaults to the user's home (paths resolve via
# $HOME on the HPC cluster). Set REPFIX_ROOT/REPFIX_DATA_DIR explicitly to override.
set -e
module purge
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export REPFIX_ROOT="${REPFIX_ROOT:-$HOME/repfix}"
export REPFIX_DATA_DIR="${REPFIX_DATA_DIR:-$REPFIX_ROOT/data}"
cd "${REPFIX_ROOT}/scripts"

# SEEDS and WORKERS injected by the caller (default: 5 seeds, 16 workers).
SEEDS=${SEEDS:-5}
WORKERS=${WORKERS:-16}

"${REPFIX_ROOT}/venv/bin/python" wf_stf_ablation_multi.py \
    --seeds "${SEEDS}" --workers "${WORKERS}" \
    --out "${REPFIX_ROOT}/results/wf_stf_ablation_multiseed.json"

echo "STF_MULTISEED_DONE"
ls -la "${REPFIX_ROOT}/results/"