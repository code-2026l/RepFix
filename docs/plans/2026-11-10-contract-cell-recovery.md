# Consumer-Cell Recovery Implementation Plan

**Goal:** Test a consumer-defined compatibility loss and obtain useful-learning
confirmation on more than one task without inflating novelty or changing earlier
verdicts.

**Architecture:** Define a convex probability cell from the healthy class and
the two fixed selective consumers. Penalize the minimum KL divergence to this
cell, using an analytic weighted pooling/capping projection. Separate objective
choice from decoder capacity, first on frozen learned features and then, only
where needed, with jointly learned encoder/readout parameters. The existing
full-KL, logit-MSE, full-feature and OLS controls remain explicit.

**Tech stack:** Local Python/PyTorch/SciPy for derivation, unit checks, analysis,
plots and LaTeX; bounded CPU-only numerical jobs remotely. No paper documents
or analysis/figure scripts are staged remotely.

## Scientific motivation and prior-work boundary

The last audit found that class agreement need not retain threshold consumers.
Matching every logit also constrains quantities those consumers do not use.
For old probabilities p0, class k and old confidence c0, let [l,u] be the
confidence interval determined by the fixed training quantiles 0.2 and 0.5.
The protected cell is a subset of the simplex with p[k] >= r*p[j] for every
j != k and l' <= p[k] <= u'. Use r=exp(0.1*min(old top-two logit gap,1)).
For finite internal bounds use l'=0.9*l+0.1*c0 and u'=0.9*u+0.1*c0; endpoints
0 and 1 remain endpoints. The healthy distribution is feasible. Unique old
classes and buffered upper boundaries avoid changing hard consumer outputs
on the closed projected cell; old ties and exact quantile ties are counted.

Define loss min_{p in cell} KL(p || softmax(z)). Its envelope gradient is
softmax(z)-p*, bounded coordinatewise. The intended contribution is this
specific multi-consumer geometry and its tested utility/compatibility behavior,
not KL projection, distillation, confidence, negative flips or set-valued targets
in general. Cite label relaxation (Lienen & Huellermeier, AAAI 2021), knowledge
distillation, positive-congruent training and backward-compatible alignment.
An independent constrained optimizer and finite differences must check the
closed form before training. No novelty claim is made merely from not finding
an identical title in search.

## Stage A: fixed-feature objective/capacity audit

Use the existing development seeds 800--802, HAR and RadioML, with their saved
healthy encoders/Adam states and original splits. No official test scoring.
The HAR donor is full-feature alignment lambda 1 from the already published
grid; the SNR donor is ordinary joint learning. These choices are based on
known development utility and are retrospective, not new confirmations.

Fit reader inputs using fitting features only (fixed mean/std). All readers
start from the fixed training-only OLS teacher-logit solution, with nonlinear
correction initialized to zero. Architecture grid:

- affine;
- diagonal quadratic expansion (linear plus centered squared standardized features);
- affine plus a 32-unit ReLU residual reader.

Each architecture uses (i) full KL at temperature 1, (ii) full KL at 2 with T^2
scaling, (iii) full KL at 4 with T^2 scaling, and (iv) the consumer-cell loss.
All use Adam lr 0.001, weight decay 0.0001, batch 1024, 60 epochs, the same
future reader minibatches (seed+41000), gradient norm cap 5, no early stopping
or validation checkpoint selection. Raw OLS-float32 remains a control. Model
task heads/encoders do not update. Cache donor features once per seed.

Feasibility is empirical disagreement <=0.025 for all three consumers on
every development seed, mean HAR primary difference to detachment >=-0.005
and reconstruction ratio <=0.95; SNR primary gain over detachment has paired
95% lower endpoint >0 and mean difference to ordinary joint >=-0.005.
Choose the smallest feasible reader architecture in the listed order, breaking
ties by mean worst-consumer disagreement. Choose full-KL temperature using
that same criterion within the selected architecture. All outcomes remain.
Primary gains are inherited from donors and cannot be attributed to a new
readout objective. If a domain passes, freeze its Stage A recipe; otherwise
run its fixed Stage B grid. Capacity and objective effects are separate claims.

## Stage B: joint consumer-cell guidance where Stage A fails

Reuse the same saved healthy checkpoints/Adam states, primary label budgets,
future encoder minibatches, original 30 HAR / 25 SNR epochs and doses 100 / 1.
Legacy input normalization is fixed to healthy fitting feature statistics.
All heads, including controls, use this same conditioning. Affine and residual
32 readers initialize to the healthy function (nonlinear output zero).

Grid: consumer-cell and full KL (T=1), affine and residual-32 heads, coefficients
0.1, 1, 10, 100. All branches use Adam lr 0.001 and a global gradient norm cap
5. Include paired detached/joint controls with the same cap, a logit-MSE
control at prior HAR 0.1 / SNR 10, and full-feature alignment at prior HAR 1 /
SNR 100. Use original warm-state optimizer moments for encoder/task heads and
new moments for additional parameters. No coefficient/epoch search follows.

Apply the same three-consumer feasibility criterion as Stage A, with task
criteria measured against the matched Stage B controls. Select the smallest
feasible architecture, then smallest feasible coefficient. Accuracy cannot
rank feasible candidates. Compare the selected cell arm with the matched
architecture's best feasible full-KL coefficient, or its lowest-disagreement
coefficient if none is feasible. Retain all failed arms. A marginally positive
mean cannot open confirmation if its stated gate fails.

## Independent training confirmation

Only passing domain/route recipes advance. Freeze selections in a versioned JSON
and publish its hash before seeds 2300--2309. These are new training seeds on
the existing official benchmark split, not a new dataset/population split.
Stage A confirmation trains the matched D/J controls and the fixed HAR donor
if required, then fits the chosen cell and temperature-KL readers on the same
donor. Stage B confirmation trains paired D/J, selected cell and the matched
KL comparator. Include OLS and the prior fixed full-feature control as appropriate.

The fixed monitor has 1024 uniformly sampled inputs, RNG 9600001+seed
(+100000 for HAR), tail 0.05/600 for a conservative 600-contract bank. No monitor
membership enters training or selection. For a complete target every seed
must have observed disagreement and finite-corpus remainder bound <=0.05 for
all three consumers. HAR reconstruction-ratio 95% upper endpoint <=0.95 and
primary paired 95% lower endpoint >=-0.005. SNR primary-gain 95% lower >0 and
difference-to-joint lower >=-0.005. Require one encoder and measured native
float32 current+legacy serving ratios <=1.20 at batch 1 and <=1.10 at 1024.
Report failures against these original cost limits without changing them.

To support an objective-specific empirical advantage, the paired 95% interval
for KL-comparator minus cell worst-consumer disagreement must have positive
lower endpoint. A gain caused only by reader capacity, normalization or full KL
is labeled accordingly. Classification-only or threshold-only successes remain
partial outcomes, not a complete target. Additional monitor sizes cannot rescue
the primary criterion. Independently reload all confirmed models/readers and
recompute metrics, probability cells, decision flows and finite-corpus bounds.

## Stronger-backbone follow-through

If either new route passes its development gate, perform one additional fixed
distinct-consumer visual pilot, seed 2400, using the existing CIFAR-100 cache,
ImageNet-V1 ResNet-18 at 64x64, 20% fine labels, healthy coarse reader, frozen
BN buffers and the previous fitting/anchor/validation split construction.
Use three epochs, SGD momentum 0.9, weight decay 0.0005, encoder lr 0.0001,
head lr 0.001, batch 128, gradient cap 5 and identical future minibatches.
F is frozen-encoder current-head learning; U is full ordinary updating. Compare
the selected cell route and matched KL route with the same healthy normalization
and decoder capacity. Joint routes use the coefficient selected on the prior
passing domain (HAR if both); frozen-feature routes fit at U's endpoint. This
is a separate stable-recipe pilot, not a continuation or retuning of the stopped
previous visual study. No auxiliary loss is added and no auxiliary-benefit or
foundation-model claim can follow from it.

Pilot utility gate: U-F >=0.02 fine accuracy and cell current >=U-0.005 and
>=F+0.02; all three old consumer changes <=0.025. If it passes, repeat at
2401--2402 with no tuning; only then freeze the recipe for five confirmation
seeds 2410--2414 and official-test scoring, using the same five-point
classification/selective bounds and noninferiority rules. If it fails, stop
visual expansion. The previous failed vision recipe and never-run arms remain.

## Resource and implementation tasks

1. Create `consumer_cells.py` and independent optimizer/gradient tests. Verify
   teacher feasibility, projection optimality, buffered consumer outputs,
   shift invariance, gradients and extreme logits locally.
2. Create `cell_recovery_cpu.py` and fixed local selection/summary tools;
   implement smoke and Stage A first. Implement Stage B only if its condition
   holds. Record exact donor/reference/split/source/order hashes and native dtypes.
3. Implement fresh confirmation, independent replay and isolated serving audit
   for frozen choices. Stage shared teacher features once and release memory.
4. Analyze locally, distinguish known-method gains from objective/capacity
   effects, and update paper/theory/figures only from complete records.
5. Rebuild and verify source/supplement/submission packages locally; update the
   existing GitHub draft and one clear latest folder. Preserve prior releases.

All remote roots are opaque random identifiers. At most two CPU workers,
four threads per small-model job, eight per visual job, 8 GiB RSS/job and
24 GiB aggregate. Per-job wall <=1800s small-model / <=2400s vision,
CPU <=7200s / <=19200s, campaign <=7200s, this round's disk <=4 GiB.
CPU caches/datasets may be reused read-only; no GPU allocation. Pull useful
JSON/logs promptly and saved states on completion, verifying SHA-256. Stop
nonfinite branches, stop queues after two failures and inspect before any
software-only restart. Do not change scientific choices to repair a failed
outcome. No tests are scored when their development gate is closed.
