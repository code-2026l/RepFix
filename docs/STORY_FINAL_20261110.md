# Final argument and English revision

The paper's central argument is now explicit: task decodability can survive an
encoder update that breaks an existing readout. Scale interventions identify one
mechanism. The deployed output rule determines what a repair must retain, and
learning capacity plus shared computation determine the cost of retention.

The introduction proceeds from the maintenance problem to the matched failure,
its mechanism, the distinction between class and threshold outputs, and the
constructive preservation results. The abstract follows that sequence. The
conclusion answers the same question rather than listing every experiment.

The signed-update result remains in the main text. The unsigned proxy definition,
calibration scans and detailed protocol controls sit beside their appendix
derivations. The preservation section separates input/readout repair, retention
of the healthy function, and measured learning/cost outcomes. The existing
disagreement inequality now introduces the practical consumer check in the main
text. Related work begins with backward compatibility and monitor maintenance.

The language pass favors specific subjects and verbs, states evidence before its
interpretation, and gives scope at the point where it affects that interpretation.
There are no bold sentence-length paragraph labels. Negative results, technical
exclusions, data access and the shared-pretraining condition remain explicit.
The change improves presentation; it does not enlarge the scientific claims.

Three primary papers informed the argument structure, without borrowing wording:

- [Vision Transformers Need Registers](https://arxiv.org/html/2309.16588):
  develop a concrete anomaly into a tested mechanism and intervention.
- [Learning Dynamics of LLM Finetuning](https://arxiv.org/html/2407.10490):
  introduce the analytic object before using it to explain empirical behavior.
- [Safety Alignment Should Be Made More Than Just a Few Tokens Deep](https://arxiv.org/html/2406.05946):
  connect a localized failure to targeted preservation experiments.

Their award status was checked in the official
[2024](https://blog.iclr.cc/2024/05/06/iclr-2024-outstanding-paper-awards/) and
[2025](https://blog.iclr.cc/2025/04/22/announcing-the-outstanding-paper-awards-at-iclr-2025/)
announcements. These are writing references, not support for RepFix's findings.

The manuscript has eight main pages and 93 pages including references and
appendices. `STORY_INTEGRITY_20261110.json` compares it with scientific release
`569632c`: all 39 numbered equations, 28 mathematical statements, 22 proof bodies,
68 table bodies, generated numerical inputs, 2,564 empirical files, citations and
27 figure exports are unchanged. Some material moves between the main text and
appendices. No numerical experiment is run in this editorial pass.

The remaining research limits are unchanged: the real visual auxiliary screens
fail, broader pretraining generalization is untested, and novelty must be assessed
against directly related work on staleness, alignment and backward compatibility.
The four-reference ViT transfer result and the original measured serving results
retain their respective data and implementation scopes.
