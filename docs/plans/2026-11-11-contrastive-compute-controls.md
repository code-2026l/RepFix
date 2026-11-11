# Cross-view training-compute controls

The primary protocol compares matched optimizer steps and label access, not
matched training FLOPs: joint evaluates two views per batch, D one. Before any
confirmation outcomes, add two standard supervised controls. Do not change the
primary development choice or its confirmatory gate after viewing these controls.

Reference6200/seed6300 pilot: (1) D2 computes CE on both complementary views,
averages the two primary losses,10epochs; (2) D20 keeps the original one-view
objective and order for20epochs. Head L2 stays identical, and labeled examples,
normalization, optimizer rates, batch128 and final reader fitting are unchanged.
Neither uses new annotations. D20's first10epochs reproduce D's update sequence.
The two controls roughly match the number of suffix forwards/backwards of a
10epoch joint run; report actual elapsed times instead of claiming exact FLOPs
or wall-clock equality. Both evaluate the same fixed held-out contrastive loss.

Retain both pilot outcomes. If the selected joint weight is no better than
either supervised control, the original matched-step result may still be
reported, but do not claim compute-efficient auxiliary learning. Stronger-control
confirmation, if needed for such a claim, must retain both controls across the
four-reference, three-continuation matrix; no choosing a favorable comparator.
This is a separately declared robustness addition, not an alteration of the
already committed primary protocol. No official test data or GPU.
