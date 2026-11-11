# Host-reserve execution after the user's memory clarification

The first full-input ViT replay was stopped by the authors' conservative 8 GiB
per-job guard at 8.15 GiB RSS, rather than by host memory exhaustion. Inspection
shows approximately 3.25 TB physical RAM and 2.74 TB available; address-space
and data resource limits are unlimited. The user explicitly requests that this
artificial per-job ceiling be removed while avoiding host exhaustion.

Replace fixed per-job/aggregate RSS ceilings for the remaining untimed replay
with a 256 GiB host-available reserve and continuous monitoring. Use at most
four 12-thread workers after checking CPU/load headroom. Keep the same numerical
programs, model weights, input corpus, batch sizes, precision, monitors, gates
and already completed isolated timing. No inference implementation is selected
again. Retain the original memory-stop record and all completed outputs.

A local-authored resource handover pauses only the registered old supervisor,
monitors its owned numerical children against the host reserve, lets them finish,
then stops that old queue before pending jobs launch. A new opaque queue runs
only missing endpoints, including the guard-stopped endpoint. Reuse an existing
immutable canonical-prefix cache only when its unchanged reference/source/data
fingerprints match. Actual chosen-code replay still evaluates every test image
at both registered batch sizes. Maintain the CPU-time/wall-time and result-storage
checks. All writing, analysis, plotting and LaTeX remain local; GPU use stays off.
