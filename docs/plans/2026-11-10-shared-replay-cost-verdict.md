# Shared replay: development cost verdict before confirmation

The fixed development utility gate passed for HAR (k=4,r=4) and SNR
(k=16,r=8). Every observed classification and selective decision was
unchanged. These are development results, not confirmation.

The isolated float32 cost screen failed the registered 1.20 / 1.10 limits.
Across both arms and three development seeds, packed TorchScript serving
had median ratios 2.198 / 1.815 for HAR and 2.544 / 2.072 for SNR at
batch 1 / 1024. The separately recorded fixed-shape Inductor pilot on the
joint seed-800 states gave 1.442 / 1.462 and 1.503 / 1.732, respectively;
its setup time and absolute latency are recorded, and it is not faster
than TorchScript current-only serving at batch 1. Neither compilation
nor analytic MAC savings establish the complete deployment target.

Retain the original choices and conduct the already specified fresh
2500--2509 confirmation for utility and compatibility as a **partial
frontier**, explicitly carrying the failed cost criterion. Do not call
this a complete positive deployment result. Both ordinary full-update
controls remain mandatory. The fixed ResNet-18 development screen may
proceed because its precondition was a small-model utility gate, not a
latency pass. No threshold, rank, seed, recipe or confirmation criterion
is relaxed here. The class-cell loss route stays stopped before testing.
