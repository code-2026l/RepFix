# Contribution boundary and the strongest defensible story

The strongest thread is the survival of a specified decision rule during useful
representation learning. Decodability, aggregate accuracy and a small distillation
loss do not establish that survival. Scale repair isolates one cause; exact replay
and measured serving programs test what it costs to retain the function.

Freezing an old path and learning another is established. Progressive Networks
retain old capabilities through frozen columns and lateral connections
([Rusu et al.](https://arxiv.org/abs/1606.04671)); Side-Tuning learns an additive
side network beside an unchanged base
([Zhang et al.](https://arxiv.org/abs/1912.13503)); LST learns from intermediate
backbone activations without backpropagating through the backbone
([Sung et al.](https://arxiv.org/abs/2206.06522)). Prefix sharing and a frozen
healthy suffix belong to this family. A visual auxiliary gain would establish
utility in this setting, not invent these architectures.

The technical construction to foreground is the same-input, two-version exact
replay under specified changes. Its useful distinction is the work reused by
the two versions. Parameter rank alone need not bound activation-difference
rank; row support makes a particular correction cheap, while dense low-rank
updates buy capacity at a different measured cost. The existing negative
latency results prevent a universal computational-saving claim.

The current text cites LST while saying that side-tuning is used as a baseline.
The implemented nonlinear head and rich-feature probe are specific matched
controls, not a full reproduction of the LST architecture. Name their actual
forms and cite both side-network families as related work. Do not imply a
head-only control exhausts the prior method's capacity.

Cross-view NT-Xent is standard contrastive learning
([Chen et al.](https://arxiv.org/abs/2002.05709)). The current screen tests whether
it supplies a real auxiliary benefit that reconstruction did not. A positive
development mean is not independent confirmation. The four fitted references
share one ImageNet-pretrained checkpoint; they are four reference fits, not four
independently pretrained backbones or four datasets.

Gaussian residual propagation, PPI and survey correction are established.
The newly completed audit does not improve exact-certificate query cost.
Marginal-preserving permutations isolate missing coupling information but are
constructed controls. Keep these findings in the evidence record or a compact
appendix rather than making them another competing headline.

A concise claim to test and write around is:

> An update can improve its task while breaking an existing decision rule.
> We isolate a scale mechanism, derive an exact replay construction for
> specified updates, and measure the utility and serving cost of retaining
> the old function.

Add a confirmed visual auxiliary benefit and its measured cost only after the
fixed gates and native replay pass. The remaining novelty question is whether
the replay analysis and consumer-specific evidence change how practitioners
evaluate or implement upgrades. Adding standard losses or renaming a frozen
branch does not answer that question.

This is an integration note, not an additional manuscript section. It preserves
concurrent edits and distinguishes known methods from the contribution being
tested.
