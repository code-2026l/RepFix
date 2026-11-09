"""Rotation semantics and frozen BatchNorm state checks, without data downloads."""
import unittest

import torch
from torch import nn

from vision_chart_cpu import bn_state, freeze_bn, rotated_batch


class VisionProtocolTests(unittest.TestCase):
    def test_rotation_labels_match_transforms(self):
        x = torch.arange(4*3*4*4).reshape(4, 3, 4, 4).float()
        labels = torch.arange(4)
        rotated = rotated_batch(x, labels)
        for k in range(4):
            torch.testing.assert_close(rotated[k], torch.rot90(x[k], k, (-2, -1)))

    def test_bn_buffers_frozen_but_affine_can_learn(self):
        network = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        network.train()
        freeze_bn(network)
        before = {key:value.clone() for key, value in bn_state(network).items()}
        network(torch.randn(8, 3, 5, 5)).square().mean().backward()
        for key, value in bn_state(network).items():
            torch.testing.assert_close(before[key], value, atol=0, rtol=0)
        self.assertGreater(float(network[1].weight.grad.norm()), 0.)


if __name__=='__main__':
    unittest.main()
