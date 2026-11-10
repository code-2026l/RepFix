# Primary readout audit after the fixed late-map pilot

The three late-map screens failed their current-head utility gates. The frozen
reference heads were fitted with standardized, L2-regularized LBFGS; continuation
used CE/AdamW, whose weight decay is not the same head regularizer. Current-head
failure therefore need not identify lost or unimproved features. This audit
tests that distinction on development only before launching any further test.

On every saved seed-3800 backbone endpoint, fit a fresh primary head using only
the same 20% labeled fitting examples and the original fit_head normalization,
LBFGS maximum 200 and L2=1e-4. No encoder update or new labels. Evaluate U,r16,
r64,r128 on their saved current features. F is the healthy linear fit; H remains
the stronger frozen concatenated-feature probe. For N, refit a linear primary
reader on concatenated healthy features and its saved 128 GELU hidden features;
this fits the same residual-head function class and supplies the matched
strong nonlinear baseline. Preserve the original online-head outcomes.

All six fits are retained. Keep the same two-point net-gain threshold against
healthy/F and half-point noninferiority against U,N,H, and all old contracts.
If no candidate passes, stop the readout-only rescue before test. If any passes,
declare its fixed complete recipe before further development/confirmation;
do not upgrade these retrospective development estimates to confirmation.
Numerical work only on remote CPUs under the existing 24-thread/24-GiB guard;
all source authoring, selection and statistical interpretation stay local.
