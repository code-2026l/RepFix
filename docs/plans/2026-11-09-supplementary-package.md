# Supplementary-material consistency and reproduction pass

The supplementary archive must accompany the polished 62-page manuscript,
including its nine main pages. The older submission ZIPs remain unchanged.

1. Compare the old supplementary ZIPs and the current source/evidence archive.
2. Replace stale entry-point documentation and supply a claim-to-evidence index.
3. State review dependencies, dataset paths, and the boundary between archived
   numerical checks and data/checkpoint-dependent retraining or state replay.
4. Preserve complete results, fixed protocols, frozen choices, negative controls,
   the arithmetic correction, and the exact delivered manuscript and figures.
5. Build a single-root supplementary archive with a file hash manifest. Exclude
   only redundant completed-run progress snapshots and editorial handoff notes.
6. From a clean extracted archive, check hashes and record provenance, run the
   focused structural tests, regenerate all 17 included figures, and rebuild the
   manuscript. Compare numerical artists and rendered pages with the delivered PDF.
7. Keep computation local and on CPU. Do not push or submit anything.
