# logs/ — job-queue receipts from the runs behind the shipped results

These are the stdout logs the training runs produced. They are here so a reader
can check that the grids were actually run, and at what cost, without having to
re-run them: each `runner` ledger records the job list it was given, every
launch with its pid and queue position, every exit code, and the wall-clock per
cell.

They are **not** the numbers. The numbers are the per-seed JSON files under
`results/` and `repro/results/`, and `results/README.md` maps each of those to
the table or figure it backs. A log is evidence about the *process*; the JSON is
the result.

## Layout

```
runner/   one ledger per job queue: `<campaign>_runner.log`, plus the
          `*_launch.log` / `*_chain.log` wrappers where a campaign was started
          by a shell script rather than by the runner directly
cert/     stdout of the finite-batch certificate runs (Proposition 6/7): the
          per-batch-size table of the two probe estimators and their envelopes
smoke/    stdout of `run_smoke.sh`, the reference output for the smoke test
```

A ledger reads:

```
[09:33:21] runner start: 50 jobs, 10 slots
[09:33:21] LAUNCH A1_har_recon_s0    pid=2539485 slot=1/10 queue=49
...
[09:36:11] DONE   A1_radioml_reg_s9  rc=0 0.8min
[09:36:11] runner FINISHED: 50/50 ok
```

## What ships and what does not

Every queue ledger still on disk ships, including the ledgers of earlier
development grids whose result files were superseded and are documented as such
in `results/README.md`. Keeping the whole receipt chain is the point: a partial
set would raise the question of what was left out. The per-cell stdout of
individual jobs (`results/<campaign>/<cell>.log`, about 1100 more files) is
**not** shipped — it duplicates the JSON beside it, and every cell's exit status
is already in its ledger.

Two substitutions were applied on export so the logs carry no deployment path.
The cluster checkout lived under an account home directory, written here as
`/home/<account>`:

| on the cluster | in this release |
|---|---|
| `/home/<account>/repfix` | `<repo>` |
| `/home/<account>` | `<home>` |

Nothing else was edited: the timestamps, pids, exit codes and numbers are as
written. The logs were scanned for the account name, the author machine's
identifiers and the cluster address, and contain none.

## Smoke test

`run_smoke.sh` at the repository root runs one unified-protocol cell (HAR,
`m=32`, eight epochs, one seed) end to end — about six seconds on CPU — and
prints each mode's collapse rate and relative scatter. `logs/smoke/run_smoke.log`
is its reference output: `joint` collapses (relative scatter near 0) while
`stf0`, `stfhard`, `stfsoft` and `stfcal` do not. The cell is small enough to
run anywhere and long enough for the healthy single-task anchor to settle, which
the STF-hard and STF-soft thresholds are anchored to; at two epochs the anchor
has not converged and those two gates misfire, so do not shorten it below eight.
