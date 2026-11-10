# Residual-block screen: technical initialization correction

The first seed-2600 process stopped before fitting any continuation branch
or saving any scientific outcome. PyTorch's BN-folding helper constructed
the effective bias of a bias-free convolution with `requires_grad=False`.
The declared support rule differentiates the effective first-map weight
and bias, so its autograd call stopped. The traceback is retained.

Explicitly enable gradients on the copied effective convolutions used
for the support calculation and on the ordinary dense last-block control.
The frozen healthy convolutions remain frozen in the replay architecture;
its declared sparse and second-map bias updates were already trainable.
This specifies the dense control in the same effective affine coordinates.
No outcome, hyperparameter, epoch, seed or gate changed. Relaunch the same
seed in a new opaque directory; retain both source versions and ledgers.
