# Bridges-2 (PSC) access for this repo

Everything needed to reach the cluster without asking. Adapted from the same
setup in `representation_ai_safety`; allocation and login are shared.

## Connection

| | |
|---|---|
| Username | `mramchandran` |
| Login node | `bridges2.psc.edu` |
| Transfer node | `data.bridges2.psc.edu` (use for `scp`/`rsync`) |
| Allocation | `mth260013p` |
| `$PROJECT` | `/ocean/projects/mth260013p/mramchandran` |
| This repo on PSC | `$PROJECT/High_Dim` |

Key-based auth works from this laptop — no password or 2FA prompt:

```bash
ssh mramchandran@bridges2.psc.edu
```

There is also a browser route at https://ondemand.bridges2.psc.edu (JupyterLab).

## Where the supplemental simulation lives

```
$PROJECT/High_Dim/supplemental/
├── results.csv        # raw per-(config, p, rep, estimator, method) rows — ~144 MB, 1.65M lines
└── tables.txt         # the only deliverable                   (make_tables.py)
```

**`results.csv` stays on PSC.** It is the only source for anything needing
per-replicate values (confidence intervals, paired tests, raw MSEs). Pull back
`tables.txt` instead.

`tables.txt` holds ten tables — the number is the scenario, the letter is the
estimator (`a` = ridgeless, `b` = ridge):

| | Scenario | |
|---|---|---|
| T1 | covariance structure × k, equal noise variance | T1a / T1b |
| T2 | covariance structure × k, 16× spread in σ² | T2a / T2b |
| T3 | the two interpolation peaks (identity) | T3a / T3b |
| T4 | ensembling scheme × noise (identity) | T4a / T4b |
| T5 | shared vs opposite-sign β (identity) | T5a / T5b |

Every cell is a **raw** Ensemble/Pooled test-MSE ratio (>1 means pooling wins)
with a 95% t confidence interval across replicates. Nothing is log-scaled and
there are no figures — the tables are the whole output.

## Everyday usage

Environment (same conda env as the other repo):

```bash
module load anaconda3 && source activate representation
```

Regenerate the tables from existing results — reads `results.csv`, runs no
simulation, takes a couple of minutes on a login node:

```bash
ssh mramchandran@bridges2.psc.edu 'cd $PROJECT/High_Dim/supplemental && module load anaconda3 && source activate representation && python make_tables.py'
```

Push a changed script up:

```bash
scp supplemental/make_tables.py mramchandran@data.bridges2.psc.edu:/ocean/projects/mth260013p/mramchandran/High_Dim/supplemental/
```

Pull derived results back (small files only):

```bash
rsync -av mramchandran@data.bridges2.psc.edu:/ocean/projects/mth260013p/mramchandran/High_Dim/supplemental/tables.txt supplemental/
```

Re-running the simulation itself (only if the grid changes — completed cells
are skipped on resume, so short `--time` limits are safe):

```bash
cd $PROJECT/High_Dim/supplemental && ./submit_all.sh
```

Monitor with `squeue -u mramchandran`; cancel with `scancel <id>`.

## Gotchas

- `$HOME` has a 25 GB quota; keep data and conda envs on `$PROJECT`.
- Login nodes have no GPUs and shouldn't run heavy work. Pandas passes over
  `results.csv` are fine; anything longer belongs in `sbatch`.
- Code flows through git; only results need `scp`/`rsync`.
