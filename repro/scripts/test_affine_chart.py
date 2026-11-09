"""Numerical identities and a non-identifiability counterexample for chart repair."""
import unittest

import torch
from torch import nn

from affine_chart import fold_chart, rebase_linear, stabilize
from compiled_readout import TrainableInterface, interface_shapes


class ChartTests(unittest.TestCase):
    def test_nonlinear_encoder_fold_and_current_head_identity(self):
        torch.manual_seed(321)
        base = nn.Module()
        base.encoder = nn.Sequential(nn.Linear(7, 11), nn.ReLU(), nn.Linear(11, 5))
        base.primary = nn.Linear(5, 3)
        base.aux = nn.Sequential(nn.Linear(5, 9), nn.ReLU(), nn.Linear(9, 7))
        model = TrainableInterface(base)
        chart = stabilize(torch.randn(5, 5, dtype=torch.float64), torch.randn(1, 5))
        deployed = fold_chart(model, chart)
        x = torch.randn(29, 7)
        h = model(x)
        emitted = deployed(x)
        torch.testing.assert_close(emitted, h @ chart['matrix'] + chart['offset'], atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(deployed.primary(emitted),
                                   torch.nn.functional.linear(h, model.primary.weight.double(),
                                                              model.primary.bias.double()), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(deployed.aux(emitted), model.aux.double()(h), atol=1e-10, rtol=1e-10)
        self.assertEqual(interface_shapes(model), interface_shapes(deployed))
        self.assertTrue(all(p.requires_grad for p in model.trunk.parameters()))

    def test_condition_floor_prevents_singular_head_rebase(self):
        chart = stabilize(torch.diag(torch.tensor([1., 0., 1e-12])), torch.zeros(3))
        self.assertEqual(chart['clipped_directions'], 2)
        self.assertAlmostEqual(chart['condition_number'], 1000.)
        head = nn.Linear(3, 2).double()
        h = torch.randn(30, 3, dtype=torch.float64)
        torch.testing.assert_close(rebase_linear(head, chart)(h @ chart['matrix']), head(h),
                                   atol=1e-10, rtol=1e-10)
        with self.assertRaises(ValueError):
            stabilize(torch.zeros(3, 3), torch.zeros(3))

    def test_identical_marginals_do_not_identify_semantics(self):
        # Isotropic four-point support: a 90-degree rotation preserves mean AND
        # covariance exactly, but changes half the old classifier's predictions.
        h = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]], dtype=torch.float64)
        rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float64)
        shifted = h @ rotation
        torch.testing.assert_close(h.mean(0), shifted.mean(0))
        torch.testing.assert_close(h.T @ h, shifted.T @ shifted)
        self.assertEqual(int(((h[:, 0] > 0) != (shifted[:, 0] > 0)).sum()), 2)
        recovered = shifted @ rotation.T
        torch.testing.assert_close(recovered, h)


if __name__ == '__main__':
    unittest.main()
