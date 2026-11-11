# Cross-checkpoint replication of the fixed visual auxiliary

Use the published AugReg ViT-B/32 checkpoint, pretrained on ImageNet-21k
and fine-tuned on ImageNet-1k. This keeps the block shape, patch size,
training examples, two views, input tensors and continuation recipe from
the original experiment. The comparison changes the pretrained weights.
Both conditions use the original input normalization to hold the input
tensors fixed. Convert the public timm checkpoint to the torchvision
parameter layout and check the exported CLS features against the unsplit
model before training.

Run references 6201–6204 with continuation seed 6310, fixed from the
previous factorial grid. For each reference, fit new heads on the same
examples as before and train a matched pair with auxiliary weights 0 and
0.1. Keep ten epochs, the last two trainable blocks, the 20% label mask,
the original learning rates and the original final reader fitting. Use
only the 5,000 validation images for evaluation. The reference means are
the four reference-level units for the paired interval. This design probes
one additional checkpoint within the ViT-B/32 architecture.

The fixed success criteria are a positive lower 95% paired interval
endpoint for fine-label accuracy gain and an upper endpoint below one
for the held-out contrastive loss ratio. Recompute the old FP32 logits
from the published checkpoint on every validation image and compare
class and both threshold decisions with the protected output. Also
evaluate the unchanged old head applied directly to the updated features.
Retain all four paired outcomes regardless of direction.

Source checkpoint SHA-256
`b0a4866d4f722434e9d26c529181f1e7072d80edcf75843f2af4e091617a4098`.
[Published model card](https://huggingface.co/timm/vit_base_patch32_224.augreg_in21k_ft_in1k)
