# Pooling Versus Ensembling for Ridge Regression Under Covariate Shift

Paper source, simulation code, results, and rebuttal materials for the NeurIPS 2026 paper by Maya Ramchandran and Rajarshi Mukherjee (OpenReview submission 31134, accepted as a poster).

**This repository is private and should stay that way as it stands.** `rebuttal/` contains confidential review text, and `simulations/rebuttal/PSC_SETUP.md` contains cluster account details. Both remain in git history even if the files are deleted later, so publish code from a fresh repository rather than by changing this one's visibility.

## Layout

| Path | Contents |
|---|---|
| `paper/submission.pdf` | The reviewed submission |
| `paper/source/` | LaTeX source and figures, exported 2026-08-04 (before any camera-ready edits) |
| `paper/presentation/` | Slides |
| `paper/earlier_versions/` | COLT 2026 draft and earlier section drafts ("federated learner" framing) |
| `simulations/main/` | Code, raw results, and figures for Figures 1 and 2 |
| `simulations/rebuttal/` | Code, raw results, and tables for the rebuttal simulations |
| `simulations/earlier_validation/` | Earlier theorem-validation scripts and results; not used for the final figures |
| `rebuttal/tables_as_posted.md` | **The tables to use in the camera-ready appendix**, with notes on what they measure |
| `rebuttal/openreview_thread.md` | Reviews, metareview, rebuttals, and comments copied from OpenReview |
| `rebuttal/hackmd_draft.md` | HackMD rebuttal draft, including the Area Chair response |
| `rebuttal/` (other files) | Earlier rebuttal drafts, review notes, and the OpenReview PDF |

## Reproducing the results

```bash
pip install -r requirements.txt
```

### Figures 1 and 2

Run these from `simulations/main/`.

| Paper output | Command | Writes |
|---|---|---|
| Figure 1 | `python make_appendix_figure.py` | `appendix_figure.pdf` |
| Figure 2 data | `python run_simulations.py` | `raw_*.csv` |
| Figure 2 | `python make_final_figures.py` | `fig_combined_final.pdf` |

### Rebuttal tables

Run these from `simulations/rebuttal/`. To rebuild the tables without rerunning the simulation:

```bash
gunzip -k results.csv.gz
python make_tables.py
```

`results.csv` is 144 MB, so only the compressed copy is tracked. The original is on PSC at `$PROJECT/High_Dim/supplemental/`.

To rerun the simulation, use `python run_experiments.py`, which skips completed cells when resumed. On PSC, use `./submit_all.sh`; the cluster scripts expect to be in `$PROJECT/High_Dim/supplemental` (see `PSC_SETUP.md`). The existing results were generated with Python 3.11.15 and the package versions in `requirements.txt`.

## Differences between the code and the submission text

- **Figure 2 replicates:** `run_simulations.py` and all four `raw_*.csv` files use 50 replicates. The submission text, Figure 2 caption, and checklist say 100.
- **Ablation in Section 4:** the random-split and no-cluster results are in `simulations/main/ablation_results_comparison_tables.md`, which reports 20 trials per configuration. No script for them has been found.
- **Compute:** the submission says all experiments ran on 4 local CPU cores. `run_simulations.py` uses 5/8 of available cores, and the rebuttal simulations ran on PSC Bridges-2.
- **Rebuttal simulations:** see "What the numbers are" in `rebuttal/tables_as_posted.md`. They differ from Section 3 in test distribution, cluster mean scale, ridge grid, and ensembling scheme.
