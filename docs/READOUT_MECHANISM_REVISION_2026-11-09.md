# Readout mechanism and real-reconstruction revision

This round extends the paired CPU evidence at baseline `af46e81bd104c7b7a973d3b8345b88496e89f430`. The earlier artifact and report are preserved separately. All code was authored locally. Remote execution was CPU-only numerical calculation, with opaque run directories, at most four single-thread workers, per-job and aggregate memory limits, CPU/wall deadlines, and host-memory/disk guards. Analysis, figures, manuscript writing, compilation, and packaging were local. No GitHub push or OpenReview action was performed.

## What the new evidence resolves

The previous per-coordinate repair did not establish that global scale mattered. We reconstructed the original healthy head and compared nine fixed repairs on every archived endpoint from the original 20 paired runs (200 endpoints, 1,800 repair evaluations). Healthy and frozen/per-coordinate endpoint accuracy, NLL, and ECE reproduce within `1e-7` before any new comparisons are interpreted.

At HAR isotropic dose 3 (10 seeds), frozen accuracy is **0.453**, mean-only correction **0.813**, global scale alone **0.938**, mean plus global scale **0.944**, and per-coordinate statistics **0.947**. The global estimate uses fitting features only, without new labels or test-statistic fitting. This directly supports a strong scale-sensitive consumer channel. It does not imply pure scalar encoder dynamics: the standardized full-feature residual can remain large even when predictions recover.

The richer controls define the boundary. At HAR projected dose 3 (five seeds), global scale reaches **0.667**, per-coordinate correction **0.782**, scaled Procrustes **0.862**, and paired ridge **0.944**, versus **0.946** for the independently refitted probe. At isotropic dose 100, global scale reaches only **0.603**, per-coordinate correction **0.829**, and paired ridge **0.915**, versus **0.935** after refitting. Covariance matching is not sufficient either; all of its outcomes, including poor ones, are retained.

All repair arms are fixed post-hoc diagnostic baselines. Ridge is `1e-3` in standardized coordinates; the covariance floor is `1e-6` of the largest eigenvalue. Paired diagonal regression, Procrustes, and ridge require the old/new features of the same fitting anchors. This access is explicitly distinguished from marginal-statistic repair. These controls are not presented as new algorithms or as a reproduction of jointly trained BC-Aligner. CORAL and backward-compatible representation learning are acknowledged in the positioning.

## Real auxiliary tasks, with their own utility measured

Ten new paired training runs add five HAR reconstruction and five RadioML I/Q reconstruction seeds, each with 10 continuation branches. The HAR grid is `1,10,100` after 60 healthy epochs and 30 continuation epochs; the RadioML grid is `30,300,3000` after 25 healthy and 15 continuation epochs. The decoder is warmed on detached features. Model, decoder, Adam state, and future minibatches are shared at branching. All 100 endpoints and all negative cases are retained.

At HAR dose 100:

- Frozen accuracy is **0.826** versus **0.941** under detachment.
- The current primary head is **0.941** versus **0.943** primary-only. This does not establish a primary-task gain or an equivalence guarantee.
- Held-out reconstruction MSE is **0.765** times the dose-matched detached decoder's error; its descriptive seed-level 95% interval is **[0.732, 0.797]**.
- The paired frozen-accuracy difference is **-0.115**, with descriptive seed-level 95% interval **[-0.216, -0.014]**.
- Mean relative scatter against primary-only continuation is **0.449**; **0/5** seeds meet the operational `<0.3` collapse criterion.
- Per-coordinate repair reaches **0.863** and a refitted head **0.941**.

This provides a real-task example in which the auxiliary improves its own objective, task accuracy remains comparatively high, the scatter criterion does not fire, and the old consumer nonetheless degrades. It is more limited than proving that the auxiliary improves the primary task.

RadioML I/Q is retained as a near-null continuation control: at dose 3000, frozen accuracy is **0.301** versus **0.304** detached, current-head accuracy **0.318** versus **0.319** primary-only, and relative trace **1.061**. Its reconstruction MSE ratio is **1.007**, with a seed interval spanning one. The historical independently trained I/Q result therefore does not transfer to this protocol. The SNR head-state factorial was not needed to support these claims and was not run; the manuscript continues to state that the historical/continuation contrast does not identify which protocol difference caused the changed onset.

Auxiliary evaluation uses the declared forward filter and frozen gate in fixed 256-example batches. Raw-head MSE is also archived. This matters for STF-cal: removal of sample-varying forward inputs can substantially worsen reconstruction even when the consumer is better preserved. The full table reports that tradeoff.

## Data and numerical verification

The RadioML cache was compared read-only against the original dictionary and the loader's per-modulation 50/50 `RandomState(0)` split. All six tensors (train/test inputs, modulation labels, and SNR targets) match exactly. Each portion has 110,000 examples and 11 modulations. Source/cache and train/test-index hashes are retained. The text now distinguishes this seeded partition from a unique official split.

The original paired driver remains byte-identical, preserving the previous source hash and reproducibility. The new verification checks 110 additional model/optimizer checkpoints, all cloned states/minibatch identities, all detached endpoint identities, the reconstructed heads, and the repair/auxiliary records' source and checkpoint links. The complete paired collection now has **30 training runs, 300 continuation endpoints, and 330 model/optimizer checkpoints**; trained checkpoints stay local and are not included in the source artifact.

The new queue contains 30 jobs (10 training, 20 checkpoint repairs), followed by 11 read-only task/split checks. These accumulated 2,298.9 seconds of job wall time across parallel workers; this is an observed execution ledger, not a CPU benchmark. Peak monitored aggregate RSS was **6,245,515,264 bytes**. Nine opaque directories across both rounds contain 358,537,854 bytes. Final read-only inspection found **no active owned run process**; all **547** remote result/checkpoint/log files match local SHA-256 copies. Uploaded bundles contain Python numerical files and no article documents. Private connections and directory mappings remain under ignored `tmp/`.

## Manuscript and artifact

The abstract, introduction, Figure 2, Algorithm 1, main real-head table, related work, and conclusion now reflect the repair hierarchy and paired real-task findings. Historical separately trained real-head results remain in the appendix. New appendix sections specify all transformations, fixed constants, anchor access, task error evaluation, full grids, uncertainty, and input-cache provenance. The main table and full repair table use larger fonts after visual inspection. Conference typography and main-text limits are retained.

The local PDF has **9 main-text pages and 51 total pages**, with references starting on page 10, no unresolved references, no overfull boxes, and blank author metadata. All-page contact sheets and selected full-page/figure inspection are part of local QA. The nine mathematical/identifiability tests pass. The source artifact includes the numerical records, logs, summary/verification JSON, manuscript, figures, and regeneration scripts, but excludes raw datasets, trained states, and private operational files.

Key files:

- `paper/RepFix_ICLR2027.{tex,pdf}`, `paper/mechanism_numbers.tex`, and the generated repair/reconstruction tables.
- `repro/results/CPUCONT/mechanism_summary.json` and `mechanism_verification.json`.
- `repro/results/CPUCONT/mechanism/` and `task_validation/` for all new scientific records.
- `repro/scripts/readout_alignment.py`, `checkpoint_repair_audit.py`, `reconstruction_task_audit.py`, and `radioml_split_audit.py` for numerical calculations.
- `repro/scripts/summarize_readout_mechanism.py`, `verify_readout_mechanism.py`, and `build_cpu_paper.py` for local analysis, verification, and compilation.

These improvements strengthen mechanism isolation and causal evidence. Generalization of the paired real-task findings beyond the tested small MLPs and fixed datasets remains a material empirical boundary. No acceptance or award outcome is claimed.
