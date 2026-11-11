# Matched current graph for batch-128 serving

The previous modular autocast program traces both suffixes together.
Its trace disables the fast attention path to support reference autocast.
Use a standalone autocast reference so the current-only and protected
programs call the same compiled FP32 prefix and current suffix.

Use the existing four TorchVision confirmation endpoints at seed 6310.
Evaluate batch 128 on the same 128 images per endpoint. Compare the
current-only program, the standalone autocast reference added to that
program, and the corresponding standalone FP32 reference. Use twelve
warmups and thirty-one interleaved samples per program. Run the four jobs
serially on twelve threads after the replication queue finishes. Report
the median of each program and the paired interval over the four endpoint
ratios. Also time the prefix and the three suffix calls separately on
the same batch as a cost diagnostic.

The autocast candidate passes the original batch limit if the upper
95% interval endpoint is at most 1.10, all current classifications
match, and no old contract exceeds 0.2% disagreement against FP32
on the 5,000 validation images. Audit class and both selective decisions
using the exact batch-128 programs that were timed. The FP32 program
must preserve all three old decision rules to support an exact decision
claim. The prior batch-one result is a separate measurement.
