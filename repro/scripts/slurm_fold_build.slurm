#!/bin/bash
#SBATCH --job-name=repfix_fold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=3G
#SBATCH --time=02:00:00
#SBATCH --output=%x.%j.log
#SBATCH --error=%x.%j.err

# Portability note: data root defaults to the user's home (paths resolve via
# $HOME on the HPC cluster). Set REPFIX_ROOT/REPFIX_DATA_DIR explicitly to override.
set -e
module purge
export OMP_NUM_THREADS=16
export REPFIX_ROOT="${REPFIX_ROOT:-$HOME/repfix}"
export REPFIX_DATA_DIR="${REPFIX_DATA_DIR:-$REPFIX_ROOT/data}"
cd "${REPFIX_ROOT}/scripts"

"${REPFIX_ROOT}/venv/bin/python" wf_repro.py --stage data --folds 0 --max-folds 10 --n-stocks 0

echo "FOLD_BUILD_DONE"
ls -la "${REPFIX_DATA_DIR}/"