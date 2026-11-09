# Third-party notices

The MIT license in [`LICENSE`](LICENSE) applies to this repository except for the
files below, which retain their own terms.

- `repro/scripts/_ref_pcgrad.py` is BSD 3-Clause, Copyright (c) 2021
  Wei-Cheng Tseng, vendored verbatim from
  https://github.com/WeiChengTseng/Pytorch-PCGrad and used only as a
  differential reference for the tests.
- `paper/natbib.sty` and `paper/fancyhdr.sty` are part of the LaTeX distribution
  and remain under their respective licenses.
- `paper/iclr2027_conference.bst` is the ICLR conference bibliography style and
  remains under its own terms.

Raw datasets and pretrained model weights are not bundled. The vision study
uses [CIFAR-100](https://cave.cs.toronto.edu/kriz/cifar.html) and the torchvision
ResNet-18 `IMAGENET1K_V1` weights. Dataset access and pretrained-weight use
remain subject to their original providers' terms. Numerical summaries and
figure scripts in this artifact do not redistribute the images or weights.
