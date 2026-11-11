# Widened frozen-feature decoder control

The current nonlinear frozen-feature control has128 hidden units, whereas the
opened visual suffix has millions of trainable parameters. Before claiming that
encoder adaptation is necessary, screen a stronger ordinary decoder. This is a
baseline robustness check, not a new adaptation method or a LST reproduction.

Use reference6200/seed6300 and the same two cached healthy views,10000 fit/5000
validation split and20% label mask. Concatenate healthy final features and the
healthy final-MLP hidden state, as in rich control H. Standardize on the two-view
labeled fitting pool, with the existing SD floor. Add a GELU hidden layer of
width1024/2048/4096, a zero-initialized residual classifier, and the original
healthy fine-label affine classifier. Train10epochs on the same selected-view
orders as N, using only labeled rows, AdamW lr1e-4, wd1e-4 and head L2 1e-4.
Refit the final affine reader on the concatenation of the frozen rich features
and learned hidden activations, using the same fitting routine/label pool as N.
The healthy encoder and coarse readout remain frozen.

Retain all three widths. Extend only a width with validation accuracy at least
.7164, i.e. at least .20 points above the selected joint pilot .7144. If more
than one qualifies, use the smallest width. Additional development6301/6302
must both beat the same-seed selected joint branch; otherwise no confirmation.
Only then use the four-reference6310--6312 matrix. No actual serving or general
superiority claim without independent confirmation and native timing. A failed
screen does not establish that every flexible frozen-feature decoder is weaker.

Numerical CPU only, no official test data or new annotations. Local authorship
and existing total concurrency/host-reserve rules continue to apply.
