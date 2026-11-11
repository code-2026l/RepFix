# Technical correction: independent decoder optimizer moments

An explicit alias test shows that loading a same-device PyTorch optimizer state
without copying its tensor values can share moments across optimizer objects.
The first ResNet pilot completed with this defect; its joint outcomes are invalid
for the paired study and will not be used for scientific selection. At discovery,
only queue completion/failure status had been inspected, not its numerical scores.
The associated ViT job had failed before training because its cache was not ready.

Reissue the unchanged pilot with a deep copy of each warm optimizer state.
Verify distinct tensor storage for cloned moments and check that the warm-state
hash remains unchanged after every training branch. A regression test updates
the clone and checks that all original moments and step counts remain intact.

No data split, seed, loss, rate, dose, architecture, stopping rule or selection
criterion changes. The original source, queue, states and logs remain archived
as a technical failure. The frozen-prefix preparation does not use this optimizer
path; its completed cache remains valid and is reused without recomputation.
