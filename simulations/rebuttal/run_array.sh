#!/bin/bash
# Bridges-2 array job: one task per covariance structure, plus one task
# for the opposite-sign beta add-on. CSV resume makes the short time
# limit safe — just resubmit and completed cells are skipped.
#
#   sbatch run_array.sh
#
#SBATCH --job-name=supplemental-sims
#SBATCH --account=mth260013p
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%a.out

# ~18 core-hours total, ~2 per array task => ~20 min on 8 cores. The 1 h
# limit is deliberate headroom, not an estimate.
#
# 8 cores rather than 16 on purpose: the dominant cost is lstsq (an SVD),
# which is memory-bandwidth-bound, so parallel efficiency drops as workers
# are added. Measured ~2x on 4 cores locally. Since RM-shared charges for
# allocated core-hours, 8 well-utilized cores burn fewer SUs than 16
# half-idle ones for the same wall-clock.
set -euo pipefail

module load anaconda3
conda activate representation

# an absolute path, not $(dirname $0): Slurm runs a spooled copy of this
# script, so $0 does not point at the repo
cd "$PROJECT/High_Dim/supplemental"
mkdir -p logs

# joblib parallelizes ACROSS replicates (n_jobs = all allocated cores).
# These vars stop each worker's BLAS from ALSO spawning a thread pool
# inside a single fit -- without them it is 8 workers x 8 threads on
# 8 cores, which thrashes. This limits threads per worker, not cores used.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

STRUCTURES=(identity diag_hetero equicorr_shared_0.2 equicorr_shared_0.5 \
            equicorr_shared_0.8 equicorr_hetero latent latent_hetero nonlinear)

# each task writes its own CSV to avoid concurrent-append clashes;
# concatenated into results.csv by merge_and_tabulate.sh after all tasks finish
if [ "$SLURM_ARRAY_TASK_ID" -le 8 ]; then
    S=${STRUCTURES[$SLURM_ARRAY_TASK_ID]}
    python run_experiments.py --structures "$S" \
        --out "results_${S}.csv" --n-jobs "$SLURM_CPUS_PER_TASK"
else
    python run_experiments.py --structures identity --beta-specs opposite \
        --out "results_opposite.csv" --n-jobs "$SLURM_CPUS_PER_TASK"
fi
