#!/bin/bash
# Submit the whole pipeline with one command:
#
#   ./submit_all.sh
#
# Queues the simulation array, then queues the merge+tabulate step to start
# only once every array task has finished successfully (afterok). If any
# array task fails or times out, the merge simply never runs -- fix the
# cause and resubmit; completed cells are skipped on the rerun.
set -euo pipefail
cd "$(dirname "$0")"

ARRAY_ID=$(sbatch --parsable run_array.sh)
echo "simulations:  job $ARRAY_ID"

MERGE_ID=$(sbatch --parsable --dependency=afterok:"$ARRAY_ID" merge_and_tabulate.sh)
echo "merge + tables: job $MERGE_ID  (waits for $ARRAY_ID)"
echo
echo "watch with:  squeue -u \$USER"
echo "results in:  results.csv, tables.txt"
