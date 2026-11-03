#!/bin/bash
# Cleanup step: merge the per-structure CSVs written by run_array.sh into a
# single results.csv, then build the summary tables.
#
# Normally submitted automatically by submit_all.sh as a dependent job, but
# it is a plain script -- safe to run by hand at any point if you only want
# tables from whichever structures have finished so far.
#
#SBATCH --job-name=supplemental-merge
#SBATCH --account=mth260013p
#SBATCH --partition=RM-shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:45:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

module load anaconda3
conda activate representation

cd "${PROJECT:-.}/High_Dim/supplemental"
mkdir -p logs

shopt -s nullglob
PARTS=(results_*.csv)
if [ ${#PARTS[@]} -eq 0 ]; then
    echo "no results_*.csv found -- did the array job run?" >&2
    exit 1
fi

# keep the header from the first file only, drop the repeated header line
# that starts every subsequent file
awk 'FNR == 1 && NR != 1 { next } { print }' "${PARTS[@]}" > results.csv
echo "merged ${#PARTS[@]} files -> results.csv ($(wc -l < results.csv) lines)"

# tables.txt is the only deliverable: ten Ensemble/Pooled ratio tables with
# 95% t CIs, one per (scenario, estimator)
python make_tables.py
