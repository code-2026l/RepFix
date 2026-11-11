# Routing specificity implementation plan

**Goal:** Determine whether fixed confidence routing improves teacher retention beyond spending the same number of additional suffix evaluations at random.

**Architecture:** Reuse the twelve hashed official-test prediction tensors. For a uniformly random subset of exactly k of n inputs, each input is selected with probability k/n. Compute expected accuracy and teacher disagreement exactly, without drawing random subsets or choosing new cutoffs. Average three continuations within each fitted reference and use paired intervals across the four references.

**Tech stack:** Local Python, PyTorch and the existing statistical helpers.

This is a retrospective analysis of previously inspected test predictions, not an independently confirmed or preregistered experiment. No training, routing threshold or prediction changes. Equal subset counts match additional suffix evaluations, not measured wall time.

1. Add `repro/scripts/analyze_vit_routing_specificity.py`; check input hashes and the fixed routing identity.
2. Report targeted versus exact random expectation, teacher disagreement captured, and matched-arm auxiliary contrasts. Validate the expectation against exhaustive subsets in a small example.
3. Add the results and the distinction between old-interface preservation and current-output approximation to the existing visual appendix. State the contribution relative to distillation, side tuning and adaptive inference without claiming those tools as new.
4. Compile locally, inspect affected pages, independently replay the new summary from the numerical archive, and refresh the source, supplement, submission folder and draft PR.
